import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback, EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import mape
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood,QuantileRegression
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts import TimeSeries,concatenate

df = pd.read_csv("Data/bitcoin_daily_since_2017.csv")
series = TimeSeries.from_dataframe(df,time_col='timestamp', value_cols='close')
# convert to float32
series = series.astype(np.float32)

# create train and test splits
train_size = 0.8
train, val = series.split_after(train_size)

# scale
scaler= Scaler()
train_transformed = scaler.fit_transform(train)
val_transformed = scaler.transform(val)
series_transformed = scaler.transform(series)

plt.figure(figsize=(10, 3))
train_transformed.plot(label="train")
val_transformed.plot(label="validation")

class PatchedPruningCallback(optuna.integration.PyTorchLightningPruningCallback, Callback):
    pass


def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 3, 14)
    out_len = trial.suggest_int("out_len", 1, in_len - 1)

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 2, in_len - 1)
    num_filters = trial.suggest_int("num_filters", 8, 256)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dilation_base = trial.suggest_int("dilation_base", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 1024)
    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PatchedPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 4
    else:
        num_workers = 0

    pl_trainer_kwargs = {
        "accelerator": "auto",
        "callbacks": callbacks,
    }

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model_name = "tcn"
    model_tcn = TCNModel(
        batch_size=batch_size,
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        n_epochs=100,
        num_layers=num_layers,
        dropout=dropout,
        dilation_base=dilation_base,
        likelihood=QuantileRegression(),
        weight_norm=weight_norm,
        kernel_size=kernel_size,
        num_filters=num_filters,
        nr_epochs_val_period=1,
        random_state=0,
        save_checkpoints=True,
        model_name=model_name,
        force_reset=True,
        optimizer_kwargs={"lr": lr},
    )

    # train the model
    model_tcn.fit(
        series=train_transformed,
        val_series=val_transformed,
    )
    model_tcn = TCNModel.load_from_checkpoint(model_name="tcn", best=True)
    # reload best model over course of training
    backtest = model_tcn.historical_forecasts(
        series_transformed,
        start=val_transformed.start_time(),
        forecast_horizon=out_len,
        num_samples=500,
        stride=out_len,
        last_points_only=False,
        retrain=False,
        verbose=True,

    )
    backtest = concatenate(backtest)
    mape1 = mape(val_transformed, backtest)

    return mape1 if mape1 != np.nan else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize"     )
study.optimize(objective, n_trials=20, callbacks=[print_callback])