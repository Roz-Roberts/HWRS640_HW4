from pathlib import Path
import click

# Commands from source files
from data import *
from train import *


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
    if structure:
        data_structure(file_path)
    else:
        click.echo("Summarizing data...")
        click.echo("Rendering Plots...")
        click.echo("Plots Rendered to Output Folder...")
        click.echo(f"Output Folder: {DEFAULT_OUTPUT_DIR}")
    click.echo("-"*20)



@CLI.command("train", help="Train the model with given parameters")
@click.option("--model", default="LSTM", help="Model type *ONLY LSTM IMPLEMENTED*", show_default=True)
@click.option("--seq-len", default=30, help="Sequence length", show_default=True)
@click.option("--epochs", default=20, help="Number of epochs", show_default=True)
@click.option("--file-path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              default=DEFAULT_DATA_DIR, show_default=True)
@click.option("--just", is_flag=True,help="A flag to output the training scheme justification to the command line. Does not allow for training!")
@click.option("--horizon", default=1, show_default=True, help="Forecasting Horizon")
def train(model: str, seq_len: int, epochs: int, file_path: str, just: bool, horizon: int) -> None:
    click.echo("-"*20)
    if just:
        training_split(file_path, just)
        click.echo("-"*20)
        return


    if model == "LSTM" and seq_len == 30 and epochs == 20:
        click.echo("Training model on default parameters...")
        click.echo("-"*20)
    else:
        click.echo("Using user-defined model parameters...")

    click.echo(f"Model type: {model}")
    click.echo(f"Sequence length: {seq_len}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Forecasting Horizon: {horizon}")
    click.echo("-"*20)
    click.echo(f"Starting Training...")



@CLI.command("evaluate", help="Evaluate the model")
@click.option("--file-name", default="best_LSTM.pt", show_default=True, help="File name (from outputs folder) to evaluate a given .pt file, default best_LSTM.pt")
def evaluate(file_name: str) -> None:
    click.echo("-"*20)
    click.echo(f"Evaluating {file_name}...")
    click.echo("-"*20)


@CLI.command("plot", help="Plot A Given File, if none given plot best LSTM results")
@click.option("--file-name", default="best_LSTM.pt", show_default=True, help="File name of the model to plot, from outputs folder")
def plot(file_name: str) -> None:
    click.echo("-"*20)
    click.echo(f"Plotting {file_name}...")
    click.echo("-"*20)




if __name__ == '__main__':
    CLI()