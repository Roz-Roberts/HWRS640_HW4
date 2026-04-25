from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import click

import data, model, plotting, utils640


def evaluate_loader(model, loader, device='cpu'):
    model.eval()

    all_obs = []
    all_pred = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            all_obs.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

    obs = np.concatenate(all_obs)
    pred = np.concatenate(all_pred)

    metrics = {
        "Mean Squared Error": utils640.mse(obs, pred),
        "Root Mean Squared Error": utils640.rmse(obs, pred),
        "Mean Absolute Error": utils640.mae(obs, pred),
        "Nash-Sutcliffe Error": utils640.nse(obs, pred)
    }

    return metrics, obs, pred


def run_validation(
    file_path,
    checkpoint_path,
    output_dir,
    seq_len=30,
    horizon=1,
    batch_size=64,
    hidden_size=64,
    num_layers=1,
    dropout=0.0,
    seed=42,
    basin_ids=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, bi, ba, _ = data.load_basin_data(file_path)


    train_ds, val_ds, test_ds, train_basins, val_basins, test_basins = data.split_dataset_by_basin(
        ds,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=seed,
        shuffle=True,
    )

    input_vars = ["prcp", "tmax", "tmin", "srad", "vp"]

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data.make_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        input_vars=input_vars,
        target_var="qobs",
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        shuffle_train=False,
        normalize_x=True,
        normalize_y=False,
    )

    mod = model.LSTM_model(
        input_size=len(input_vars),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        mod.load_state_dict(checkpoint["model_state_dict"])
    else:
        mod.load_state_dict(checkpoint)

    metrics, obs, pred = evaluate_loader(mod, test_loader, device=device)

    basin_metrics_dict = plotting.plot_three_test_basins(
        model=mod,
        test_dataset=test_dataset,
        basin_ids=basin_ids,
        output_dir=output_dir,
        device=device,
        show=True,
    )

    click.echo("Per-basin test metrics:")
    for basin, metrics in basin_metrics_dict.items():
        click.echo(f"Basin {basin}")
        basin_name = bi.loc[bi["basin_id"] == basin, "basin_name"].values[0]
        click.echo(f"Basin Location: {basin_name}")
        click.echo(f"  MSE:  {metrics['MSE']:.6f}")
        click.echo(f"  RMSE: {metrics['RMSE']:.6f}")
        click.echo(f"  MAE:  {metrics['MAE']:.6f}")
        click.echo(f"  NSE:  {metrics['NSE']:.6f}")
        click.echo(f"  Bias: {metrics['Bias']:.6f}")


    return metrics