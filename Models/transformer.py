import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.metrics import mape
from darts.models import TransformerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape,mae,mse
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

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

transformer = TransformerModel(
    input_chunk_length=3,
    output_chunk_length=1,
    batch_size=410,
    n_epochs=100,
    model_name="transformer",
    nr_epochs_val_period=1,
    d_model=512,
    nhead=8,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=129,
    dropout=0.10438779126742549,
    activation="relu",
    random_state=42,
    save_checkpoints=True,
    force_reset=True,
    optimizer_kwargs={"lr": 0.0003544614845438686},
)

transformer.fit(series=train_transformed, val_series=val_transformed, verbose=True)

backtest = transformer.historical_forecasts(
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
test.plot(label="actural")
test_result.plot(label="backtest ")

print([mae(test,test_result), mse(test,test_result), mape(test,test_result)])
transformer.save("../ModelFiles/transformer.pt")