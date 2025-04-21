import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.datasets import EnergyDataset
from darts.metrics import r2_score
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape,mse,mae
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

df = pd.read_csv("Data/bitcoin_daily_since_2017.csv")
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

model_name = "nbeats_run"
model_nbeats = NBEATSModel(
    input_chunk_length=6,
    output_chunk_length=1,
    generic_architecture=True,
    num_stacks=1,
    num_blocks=3,
    num_layers=2,
    layer_widths=481,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=345,
    random_state=42,
    model_name=model_name,
    save_checkpoints=True,
    force_reset=True,
    dropout=0.43638789402466166,
    optimizer_kwargs={"lr": 0.0006902154545815396},


    **generate_torch_kwargs(),
)


model_nbeats.fit(train_transformed, val_series=val_transformed)
model_nbeats = NBEATSModel.load_from_checkpoint(model_name=model_name, best=True)
backtest = model_nbeats.historical_forecasts(
    series_transformed,
    start=test.start_time(),
    forecast_horizon=1,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=True,

)
backtest = concatenate(backtest)
test_result=scaler.inverse_transform(backtest)
plt.figure(figsize=(10, 6))
test.plot(label="actual")
test_result.plot(label="prediction ")
print([mae(test,test_result), mse(test,test_result), mape(test,test_result)])
model_nbeats.save("../ModelFiles/nbeats.pt")