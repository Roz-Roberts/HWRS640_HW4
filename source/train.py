from pathlib import Path
import click
import numpy as np
import torch
import torch.nn as nn
import json

import data, model, plotting


def nse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Function calculate the NSE (Nash-Sutcliffe Efficiency)

    NSE = 1 - sum[(Q_o(t) - Q_m(t))^2] / sum[(Q_o(t) - Q(bar)_o)^2]
        = 1 - sum[(obs - pred)^2] / sum[(obs - mean(obs))^2]

    :param y_true: True (Observed) Q_obs Values
    :param y_pred: Predicted Q_obs Values
    :return: NSE Value
    """

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if denominator.item() == 0:
        nse = np.nan
        return nse
    else:
        nse = 1.0 - (numerator / denominator)
        nse = nse.item()
        return nse


def run_single_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    """
    Run a single validation, training, or testing epoch.
    If optimizer is provided, run a training epoch specifically, otherwise evaluate.

    :param model: LSTM model from model.py
    :param loader: PyToch Dataloader from data.py
    :param criterion: The loss criterion to use for training epochs.
    :param optimizer: The optimizer for training
    :param device: Used torch device (cuda or cpu) depending on which device is present on the system.
    :return:
    """

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x,y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            y_pred = model(x)
            loss = criterion(y_pred, y)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * x.size(0)

        all_preds.append(y_pred.detach().cpu())
        all_targets.append(y.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    epoch_nse = nse_torch(all_targets, all_preds)

    return epoch_loss, epoch_nse


def save_check(model, optimizer, epoch, train_loss, val_loss, path, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    torch.save(checkpoint, path)


def load_previous_best_val_loss(best_model_path):
    if not best_model_path.exists():
        return float("inf")

    checkpoint = torch.load(best_model_path, map_location="cpu")

    if "val_loss" in checkpoint:
        return checkpoint["val_loss"]

    return float("inf")


def train_model(file_path,
        output_dir,
        seq_len=30,
        horizon=1,
        batch_size=64,
        learning_rate=1e-3,
        epochs=20,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        target_var="qobs",
        input_vars=None,
        seed=42,
        checkpoint_name="best_lstm.pt",
        nse_interval=10,
        device=None,
        ):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    np.random.seed(seed)

    # print(Path(output_dir))
    # exit("Training ERROR Hunting Exit")

    output_dir = Path(output_dir) / "training_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ds, bi, ba, vs = data.load_basin_data(file_path)

    train_ds, val_ds, test_ds, train_basins, val_basins, test_basins = data.split_dataset_by_basin(ds)

    if input_vars is None:
        input_vars = vs[0]

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = data.make_dataloaders(train_ds, val_ds, test_ds, input_vars, target_var,seq_len,horizon,batch_size)

    lstm_model = model.LSTM_model(input_size=len(input_vars),
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  output_size=1,
                                  dropout=dropout).to(device)

    # Loss and optimization
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    # Tracking the training data
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_nse": [],
        "val_nse": [],
        "config": ""
    }

    best_val_loss = float("inf")
    best_model_path = output_dir / checkpoint_name
    last_model_path = output_dir / "last_lstm.pt"

    click.echo("-" * 20)
    click.echo("Training configuration")
    click.echo("-" * 20)
    click.echo(f"Device: {device}")
    click.echo(f"Train basins: {len(train_basins)}")
    click.echo(f"Validation basins: {len(val_basins)}")
    click.echo(f"Test basins: {len(test_basins)}")
    click.echo(f"Input variables: {', '.join(input_vars)}")
    click.echo(f"Target variable: {target_var}")
    click.echo(f"Sequence length: {seq_len}")
    click.echo(f"Horizon: {horizon}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Hidden size: {hidden_size}")
    click.echo(f"Num layers: {num_layers}")
    click.echo("-" * 20)
    click.echo(f"Best Output path: {best_model_path}")
    click.echo("-" * 20)

    # training and validation loop

    for epoch in range(1, epochs + 1):
        # Training component

        best_val_loss = load_previous_best_val_loss(best_model_path)
        click.echo(f"Previous best validation loss: {best_val_loss:.6f}")

        click.echo(f"Starting Epoch {epoch}")
        train_loss, train_nse = run_single_epoch(
            model=lstm_model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # Validation component
        val_loss, val_nse = run_single_epoch(
            model=lstm_model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_nse"].append(train_nse)
        history["val_loss"].append(val_loss)
        history["val_nse"].append(val_nse)

        # Report the loss values while looping

        click.echo(f"Epoch: {epoch}/{epochs} | Train MSE: {train_loss} | Val MSE: {val_loss}")
        config = {
            "file_path": str(file_path),
            "output_dir": str(output_dir),
            "seq_len": seq_len,
            "horizon": horizon,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "target_var": target_var,
            "input_vars": input_vars,
            "seed": seed,
            "checkpoint_name": checkpoint_name,
            "nse_interval": nse_interval,
        }
        history["config"] = config

        if (epoch % nse_interval) == 0 or (epoch == epochs):
            click.echo(f"Train NSE: {train_nse} | Val NSE: {val_nse}")
        click.echo("-" * 20)
        # Best model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_check(lstm_model, optimizer, epoch, train_loss, val_loss, best_model_path, config)
            click.echo("-" * 20)
            click.echo(f"Saved new best model to: {best_model_path}")
        else:
            click.echo("Current model did not improve over saved best model.")
            run_checkpoint_path = output_dir / f"run_seq{seq_len}_hidden{hidden_size}_layers{num_layers}_lr{learning_rate}.pt"
            save_check(lstm_model, optimizer, epoch, train_loss, val_loss, run_checkpoint_path, config)


        # Last check point saving
        save_check(lstm_model, optimizer, epoch, train_loss, val_loss, best_model_path, config)


    click.echo("-" * 20)
    click.echo("Training Completed")
    click.echo(f"Best validation MSE: {best_val_loss}")
    click.echo(f"Best Model saved to: {best_model_path}")
    click.echo("-" * 20)

    history_path = output_dir / "training_history.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    click.echo(f"Saved Most Recent Training History to {history_path}")
    click.echo("This saved history will be overwritten during the next training run, move or rename the file to save it permanently.")
    click.echo("-" * 20)
    plotting.plot_training_history(history, output_dir=None, show=True)

    return {"model": model, "history": history, "best_model_path": best_model_path, "last_model_path": last_model_path, "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}
