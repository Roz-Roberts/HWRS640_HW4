import matplotlib.pyplot as plt
import click, data, random
from pathlib import Path
import torch
import numpy as np

import utils640

def exploration_plots(typ, fp, out = None):
    # We need to get the following for plots: validation_basins, training_basins, target_variable
    ds, bi, ba, _ = data.load_basin_data(fp)


    indxes = random.sample(range(len(bi)), typ)

    click.echo(f"Plotting Time-Series for Basins: {indxes}")
    basins = list(bi.iloc[indxes, :]['basin_id'])

    dta = ds.sel(basin=basins)

    x_dta = dta['time']
    fig, ax = plt.subplots(3, 1, figsize=(8,12))
    for b in basins:
        y_dta = dta.sel(basin=b)['qobs']
        ax[0].plot(x_dta, y_dta, label=f"Basin {b}")

    ax[0].set_title(f"Observed Stream Flow Time Series for {typ} Basins")
    ax[0].set_xlabel("Time (Years)")
    ax[0].set_ylabel("Observed Streamflow (mm/Day)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].hist(dta.sel(basin=basins[0])['qobs'], bins=20)
    ax[1].set_title(f"Histogram of Stream Flow For {basins[0]}")
    ax[1].set_xlabel("Bins")
    ax[1].set_ylabel("Count")
    ax[1].grid(True)

    time = dta.sel(basin=basins[-1])['time']
    qobs = dta.sel(basin=basins[-1])['qobs']
    prcp = dta.sel(basin=basins[-1])['prcp']

    ax2 = ax[2].twinx()
    line1 = ax[2].plot(time, qobs, label=f"Observed Stream Flow for Basin {basins[-1]}", color="red")
    line2 = ax2.plot(time, prcp, label=f"Precipitation for Basin {basins[-1]}", color="blue")

    ax[2].set_title(f"Time Series of Streamflow and Precipitation for {basins[-1]}")
    ax[2].set_xlabel("Time (Years)")
    ax[2].set_ylabel("Observed Streamflow (mm/day)") #, color="red")
    ax2.set_ylabel("Precipitation (mm/day)") # , color="blue")

    ax[2].grid(True)

    # Make both lines appear on one legend
    lines = [line1[0], line2[0]]
    labels = [l.get_label() for l in lines]
    ax[2].legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.subplots_adjust(hspace=0.5)
    if out != None:
        plt.savefig(f"{out}")
    else:
        plt.show(block=True)
    return

def plot_training_history(history, output_dir = None, show = True):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    best_epoch = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i]) + 1
    ax[0].axvline(best_epoch, linestyle="--", label=f"Best Epoch: {best_epoch}", color='red')
    ax[1].axvline(best_epoch, linestyle="--", label=f"Best Epoch: {best_epoch}", color='red')
    ax[2].axvline(best_epoch, linestyle="--", label=f"Best Epoch: {best_epoch}", color='red')


    # Plot 1: training loss
    ax[0].plot(epochs, history["train_loss"], marker="o", label="Training Loss", color='blue')
    ax[0].set_title("Training Loss vs. Epoch")
    # ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Training MSE Loss")
    ax[0].grid(True)
    ax[0].legend()

    # Plot 2: validation loss:
    ax[1].plot(epochs, history["val_loss"], marker="o", label="Validation Loss", color="blue")
    ax[1].set_title("Validation Loss vs. Epoch")
    ax[1].set_title("Validation Loss vs. Epoch")
    # ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Validation MSE Loss")
    ax[1].grid(True)
    ax[1].legend()

    # Plot 3: validation metric
    ax[2].plot(epochs, history["val_nse"], marker="o", label="Validation NSE", color="blue")
    ax[2].set_title("Validation NSE vs. Epoch")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Validation NSE")
    ax[2].grid(True)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return

def run_saved_plots(history_path, output_dir=None, show=True):
    history = utils640.load_history(history_path)
    plot_training_history(history, output_dir=output_dir, show=show)


def plot_three_test_basins(
    model,
    test_dataset,
    basin_ids=None,
    output_dir=None,
    device="cpu",
    show=True,
):
    model.eval()

    available_test_basins = list(dict.fromkeys([sample[0] for sample in test_dataset.samples]))

    if len(basin_ids) == 0:
        basin_ids = available_test_basins[:3]
        click.echo(f"Available Test Basins: {available_test_basins}")
    else:
        basin_ids = basin_ids[:3]

    if len(basin_ids) < 3:
        raise ValueError("Need at least three test basins to make the 2x3 validation plot.")

    fig, ax = plt.subplots(2, 3, figsize=(18, 8))

    all_metrics = {}

    for col, basin_id in enumerate(basin_ids):
        times, obs, pred = utils640.collect_basin_predictions(
            model=model,
            test_dataset=test_dataset,
            basin_id=basin_id,
            device=device,
        )

        metrics = utils640.basin_metrics(obs, pred)
        all_metrics[basin_id] = metrics

        # ---------------------------
        # Top row: time series
        # ---------------------------
        ax[0, col].plot(times, obs, label="Observed")
        ax[0, col].plot(times, pred, label="Predicted")
        ax[0, col].set_title(f"Basin {basin_id} Time Series")
        ax[0, col].set_xlabel("Time")
        ax[0, col].set_ylabel("Streamflow")
        ax[0, col].grid(True)
        ax[0, col].legend()

        # ---------------------------
        # Bottom row: parity plot
        # ---------------------------
        ax[1, col].scatter(obs, pred, alpha=0.5)

        min_val = min(np.min(obs), np.min(pred))
        max_val = max(np.max(obs), np.max(pred))

        ax[1, col].plot([min_val, max_val], [min_val, max_val], linestyle="--")

        ax[1, col].set_title(
            f"Basin {basin_id} Parity\n"
            f"NSE={metrics['NSE']:.3f}, RMSE={metrics['RMSE']:.3f}"
        )
        ax[1, col].set_xlabel("Observed Streamflow")
        ax[1, col].set_ylabel("Predicted Streamflow")
        ax[1, col].grid(True)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "Figure_2.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        click.echo(f"Saved validation plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return all_metrics