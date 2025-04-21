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
from darts.models import NBEATSModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts import TimeSeries,concatenate

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

class PatchedPruningCallback(optuna.integration.PyTorchLightningPruningCallback, Callback):
    pass


def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 3, 14)
    out_len = trial.suggest_int("out_len", 1, in_len - 1)

    # Other hyperparameters
    num_stacks = trial.suggest_int("num_stacks", 1, 10)
    num_blocks = trial.suggest_int("num_blocks", 1, 5)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    generic_architecture = trial.suggest_categorical("generic_architecture", [False, True])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    layer_widths = trial.suggest_int("layer_widths", 16, 1024)
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

    model_nbeats = NBEATSModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        generic_architecture=generic_architecture,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        n_epochs=100,
        nr_epochs_val_period=1,
        batch_size=batch_size,
        random_state=42,
        model_name="nbeats_run",
        save_checkpoints=True,
        force_reset=True,
        optimizer_kwargs={"lr": lr},
    )

    # train the model
    model_nbeats.fit(
        series=train_transformed,
        val_series=val_transformed,
    )
    model_nbeats = NBEATSModel.load_from_checkpoint(model_name="nbeats_run", best=True)
    # reload best model over course of training
    backtest = model_nbeats.historical_forecasts(
        series_transformed,
        start=test_transformed.start_time(),
        forecast_horizon=out_len,
        stride=out_len,
        last_points_only=False,
        retrain=False,
        verbose=True,

    )
    backtest = concatenate(backtest)
    test_result = scaler.inverse_transform(backtest)
    mape1 = mape(test, test_result)

    return mape1 if mape1 != np.nan else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize"     )
study.optimize(objective, n_trials=20, callbacks=[print_callback])