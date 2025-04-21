import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import EnergyDataset
from darts.models import RNNModel
from darts.timeseries import concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, mae,mse
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

df = pd.read_csv("bitcoin_daily_since_2017.csv")
series = TimeSeries.from_dataframe(df,time_col='timestamp', value_cols='close')
# convert to float32
series = series.astype(np.float32)

# create train and test splits
train_size = 0.8
train, split = series.split_after(train_size)
val,test=split.split_after(0.5)
# scale
scaler= Scaler()
train_transformed = scaler.fit_transform(train)
val_transformed = scaler.transform(val)
test_transformed = scaler.transform(test)
series_transformed = scaler.transform(series)

plt.figure(figsize=(10, 3))
train_transformed.plot(label="train")
val_transformed.plot(label="validation")
test_transformed.plot(label="test")

model_name = "Deepar"
Deeparmodel = RNNModel(
    model="LSTM",
    hidden_dim=150,
    n_rnn_layers=1,
    dropout=0.21760665462763193,
    batch_size=61,
    n_epochs=100,
    optimizer_kwargs={"lr": 0.0008867340500319594},
    random_state=42,
    training_length=25,
    input_chunk_length=3,
    output_chunk_length=1,
    likelihood=GaussianLikelihood(),
    model_name=model_name,
    save_checkpoints=True,  # store the latest and best performing epochs
    force_reset=True,
    **generate_torch_kwargs(),
)
Deeparmodel.fit(train_transformed, val_series=val_transformed)
Deeparmodel = RNNModel.load_from_checkpoint(model_name=model_name, best=True)
backtest = Deeparmodel.historical_forecasts(
    series_transformed,
    start=test.start_time(),
    forecast_horizon=1,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=True,
    num_samples=200,
)
backtest = concatenate(backtest)
test_result=scaler.inverse_transform(backtest)
plt.figure(figsize=(10, 6))
test.plot(label="true value")
test_result.plot(label="backtest ")
print([mae(test,test_result), mse(test,test_result), mape(test,test_result)])
Deeparmodel.save("../ModelFiles/deepar.pt")