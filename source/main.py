from pathlib import Path
import click, shutil

# Commands from source files
import data, model, train, validation, plotting, utils640

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR.joinpath("data")
DEFAULT_OUTPUT_DIR = PROJECT_DIR.joinpath("outputs")

@click.group()
def CLI() -> None:
    "Main Command Line Interface for HWRS640 HW 4"
    pass

@CLI.command("summarize-data", help="Summary of dataset")
@click.option("--structure",is_flag=True, help="Flag to output the structure of the Dataset and acceptable variables to use to the command line")
@click.option("--file-path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              default=DEFAULT_DATA_DIR, help="File path to the MiniCamels data file. Defaults to provided data folder.", show_default=True)
def summarize_data(structure: bool, file_path: str) -> None:
    click.echo("-" * 20)
    if file_path != DEFAULT_DATA_DIR:
        click.echo(f"You have selected a separate MiniCamels data file location at {file_path}")
        click.echo("If you encounter issues leave this field blank and the CLI will use the default data provided with this CLI.")
        click.echo("The default data contained in this CLI is the same MiniCamels data files as of: 04/24/2026")
        click.echo("-"*20)


    if structure:
        utils640.data_structure(file_path)
    else:
        click.echo("Summarizing data...")
        click.echo("Rendering Exploration Plots...")
        plotting.exploration_plots(5, file_path, f"{DEFAULT_OUTPUT_DIR}\\summary_plots.png")
        click.echo("Plots Rendered to Output Folder...")
        click.echo(f"Output Folder: {DEFAULT_OUTPUT_DIR}\\summary_plots.png")
    click.echo("-"*20)
    return



@CLI.command("train", help="Train the LSTM model with given parameters")
@click.option(
    "--file-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="Path to MiniCamels data directory",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory for checkpoints and outputs",
)
@click.option("--seq-len", default=30, type=int, show_default=True, help="Sequence length")
@click.option("--horizon", default=1, type=int, show_default=True, help="Forecast horizon")
@click.option("--learning-rate", default=1e-3, type=float, show_default=True, help="Learning rate")
@click.option("--batch-size", default=64, type=int, show_default=True, help="Batch size")
@click.option("--epochs", default=20, type=int, show_default=True, help="Number of epochs")
@click.option("--hidden-size", default=64, type=int, show_default=True, help="LSTM hidden size")
@click.option("--num-layers", default=1, type=int, show_default=True, help="Number of LSTM layers")
@click.option("--dropout", default=0.0, type=float, show_default=True, help="LSTM dropout")
@click.option("--seed", default=42, type=int, show_default=True, help="Random seed")
@click.option("--nse-interval", default=1, type=int, show_default=True, help="How often to print NSE")
@click.option("--just", is_flag=True,help="A flag to output the training scheme justification to the command line. Does not allow for training!")

def train_command(file_path,
    output_dir,
    seq_len,
    horizon,
    learning_rate,
    batch_size,
    epochs,
    hidden_size,
    num_layers,
    dropout,
    seed,
    nse_interval,
    just: bool) -> None:
    click.echo("-"*20)
    if just:
        data.training_split_justification()
        click.echo("-"*20)
        return

    click.echo("Supervised Learning Setup:")
    click.echo("-"*20)
    utils640.explain_supervised_setup(file_path, seq_len, horizon)
    click.echo("-"*20)

    if model == "LSTM" and seq_len == 30 and epochs == 20:
        click.echo("Training model on default parameters...")
        click.echo("-"*20)
    else:
        click.echo("Using user-defined model parameters...")
        click.echo("-"*20)

    click.echo(f"Starting Training...")
    train.train_model(file_path=file_path, output_dir=output_dir, seq_len=seq_len, horizon=horizon, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, seed=seed, nse_interval=nse_interval)
    return



@CLI.command("evaluate", help="Evaluate the model")
@click.option("--file-path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default=DEFAULT_DATA_DIR, show_default=True)
@click.option("--checkpoint-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--seq-len", default=30, type=int, show_default=True)
@click.option("--horizon", default=1, type=int, show_default=True)
@click.option("--batch-size", default=64, type=int, show_default=True)
@click.option("--hidden-size", default=64, type=int, show_default=True)
@click.option("--num-layers", default=1, type=int, show_default=True)
@click.option("--dropout", default=0.0, type=float, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option(
    "--basin-id",
    "basin_ids",
    multiple=True,
    help="Optional test basin ID to plot. Can be used up to three times.",
    default = None)
def evaluate(
    file_path,
    checkpoint_path,
    output_dir,
    seq_len,
    horizon,
    batch_size,
    hidden_size,
    num_layers,
    dropout,
    seed,
    basin_ids,
):
    metrics = validation.run_validation(
        file_path=file_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
        basin_ids=basin_ids,
    )
    # click.echo(f"Validation Result Metrics: {metrics}")



@CLI.command("plot", help="Plot a given file, if none given plot best LSTM results")
@click.option("--explore", is_flag=True, help="Flag to display the exploration plots")
@click.option("--history-plot", is_flag=True, help="Flag to display a training history plot, or just the latest training run plots.")
def plot(explore:bool, history_plot:bool) -> None:
    click.echo("-"*20)
    if explore:
        typ = click.prompt("Number of Stream Flow Time Series Plots To Make", default = 5, show_default=True)
        sv = click.prompt("Save Plot? Y/N", default="N", show_default=True)
        if sv == "Y":
            plotting.exploration_plots(typ, DEFAULT_DATA_DIR, f"{DEFAULT_OUTPUT_DIR}\\exploration_plots.png")
            click.echo("Plots Rendered to Output Folder...")
            click.echo(f"Output Folder: {DEFAULT_OUTPUT_DIR}\\exploration_plots.png")
        else:
            plotting.exploration_plots(typ, DEFAULT_DATA_DIR)
    elif history_plot:
        sv = click.prompt("History File Name to Plot, leave blank for default", default="training_history.json", show_default=True)
        histpath = DEFAULT_OUTPUT_DIR / "training_results" / sv
        plotting.run_saved_plots(histpath)

    click.echo("-"*20)
    return


if __name__ == '__main__':
    CLI()