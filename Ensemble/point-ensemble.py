import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape,mse,mae
from darts.models import (
    NBEATSModel,
    TransformerModel,
    RNNModel,
    RegressionEnsembleModel,
    TCNModel,
    NaiveEnsemble,
)
from darts import TimeSeries,concatenate
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

mynbeats=NBEATSModel.load("Models/nbeats.pt")
mytransformer=TransformerModel.load("Models/transformer.pt")

pretrain_ensemble = RegressionEnsembleModel(
    forecasting_models=[mynbeats,mytransformer],
    regression_train_n_points=1,
    train_forecasting_models=False,
    train_using_historical_forecasts=False,
)
# RegressionEnsemble must train the ensemble model, even if the forecasting models are already trained
pretrain_ensemble.fit(train_transformed)

backtest = pretrain_ensemble.historical_forecasts(
    series_transformed,
    start=test.start_time(),
    forecast_horizon=1,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=False,

)
backtest = concatenate(backtest)
test_result=scaler.inverse_transform(backtest)

test_result.plot()
test.plot()

print(
[mape(test,test_result),mae(test,test_result),mse(test,test_result)]
)

naive_ensemble = NaiveEnsembleModel(
    forecasting_models=[mynbeats, mytransformer],
    train_forecasting_models=False,

)
backtest = naive_ensemble.historical_forecasts(
    series_transformed,
    start=test.start_time(),
    forecast_horizon=1,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=False,

)
backtest = concatenate(backtest)

test_result=scaler.inverse_transform(backtest)
plt.figure(figsize=(10, 6))
test.plot(label="actual")
test_result.plot(label="backtest ")

print(
[mape(test,test_result),mae(test,test_result),mse(test,test_result)]
)