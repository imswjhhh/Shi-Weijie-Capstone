import warnings

import pandas as pd
import numpy as np
import darts.utils.timeseries_generation as tg
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import EnergyDataset
from darts.models import TCNModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.missing_values import fill_missing_values
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape,mse,mae
warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

import matplotlib.pyplot as plt


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

model_name = "TCN_Test"
tcnmodel = TCNModel(
    batch_size=75,
    input_chunk_length=5,
    output_chunk_length=2,
    n_epochs=100,
    num_layers=3,
    dropout=0.29864516182384304,
    dilation_base=2,
    weight_norm=True,
    kernel_size=3,
    num_filters=87,
    likelihood=QuantileRegression(),
    nr_epochs_val_period=1,
    optimizer_kwargs={"lr":0.0004341629303850734},
    random_state=42,
    save_checkpoints=True,
    model_name=model_name,
    force_reset=True,
    **generate_torch_kwargs(),
)

tcnmodel.fit(
    series=train_transformed,
    val_series=val_transformed,
)

tcnmodel = TCNModel.load_from_checkpoint(model_name=model_name, best=True)
backtest = tcnmodel.historical_forecasts(
    series=series_transformed,
    start=test_transformed.start_time(),
    forecast_horizon=2,
    stride=2,
    num_samples=500,
    last_points_only=False,
    retrain=False,
    verbose=True,
)
backtest = concatenate(backtest)
test_result=scaler.inverse_transform(backtest)
plt.figure(figsize=(10, 6))
test.plot(label="actual")
test_result.plot(label="backtest ")
plt.legend()
print([mae(test,test_result), mse(test,test_result), mape(test,test_result)])
tcnmodel.save("../ModelFiles/deeptcn.pt")