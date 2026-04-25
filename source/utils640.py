import click, json
import numpy as np
from pathlib import Path
import torch

import data

def data_structure(file_path):
    ds, bi, ba, veris = data.load_basin_data(file_path)
    click.echo("Total Dataset Structure:")
    click.echo(f"Number of Basins: {bi.shape[0]}")
    click.echo(f"Time Span of Dataset: {str(ds['time'].to_numpy()[0])[:-19]} to {str(ds['time'].to_numpy()[-1])[:-19]}")
    click.echo(f"Dynamic Input Variables: {', '.join(veris[0])}")
    click.echo(f"Target Variable: {veris[1]}")
    click.echo(f"Number of Static Attributes: {ba.shape[1]}")
    return


def explain_supervised_setup(file_path: str, seq_len:int, horizon:int) -> None:
    ds, _, _, veris = data.load_basin_data(file_path)

    click.echo(f"Sequence length: {seq_len} days")
    click.echo(f"Forecast horizon: {horizon} day(s) ahead")
    click.echo(f"Model input: previous {seq_len} days of {', '.join(veris[0])}")
    click.echo(f"Target: {veris[1]} at day t + {horizon}")
    click.echo(f"Each supervised sample uses a sliding window of {seq_len} days as input\n"
        f"and predicts the target {horizon} day(s) after the end of that window.")
    return


def load_history(history_path):
    history_path = Path(history_path)
    with open(history_path, "r") as f:
        history = json.load(f)
    return history


def mse(obs, pred):
    return np.mean((obs - pred)**2)


def rmse(obs, pred):
    return np.sqrt(mse(obs, pred))


def mae(obs, pred):
    return np.mean(np.abs(obs - pred))

def nse(obs, pred):
    numerator = np.sum((obs-pred)**2)
    denominator = np.sum((obs-np.mean(obs)))
    if denominator == 0:
        return np.nan

    return 1 - (numerator/denominator)

def basin_metrics(obs, pred):
    return {
        "MSE": mse(obs, pred),
        "RMSE": rmse(obs, pred),
        "MAE": mae(obs, pred),
        "NSE": nse(obs, pred),
        "Bias": np.mean(pred - obs),
    }

def collect_basin_predictions(model, test_dataset, basin_id, device="cpu"):
    model.eval()

    times = []
    obs_values = []
    pred_values = []

    with torch.no_grad():
        for idx, (sample_basin, t) in enumerate(test_dataset.samples):
            if sample_basin != basin_id:
                continue

            x, y = test_dataset[idx]
            x = x.unsqueeze(0).to(device)

            pred = model(x).detach().cpu().item()

            target_time_index = t + test_dataset.horizon - 1
            target_time = test_dataset.ds["time"].to_numpy()[target_time_index]

            times.append(target_time)
            obs_values.append(y.item())
            pred_values.append(pred)

    return np.array(times), np.array(obs_values), np.array(pred_values)
