import click
import sys


base_dir = "../"

sys.path.append(base_dir)


@click.group()
def CLI() -> None:
    pass

@CLI.command("summarize-data", help="Summary of best results in graphical format")
def summarize_data():
    click.echo("-"*20)
    click.echo("Summarizing data...")
    click.echo("Rendering Plots...")
    click.echo("Plots Rendered to Output Folder...")
    click.echo(f"Output Folder: {sys.path[0]}\\outputs")
    click.echo("-"*20)

@CLI.command("train", help="Train the model with given parameters: defaults: LSTM, sequence length = 30, epochs = 20")
@click.option("--model", default="LSTM", help="Model type *ONLY LSTM IMPLEMENTED*")
@click.option("--seq-len", default=30, help="Sequence length")
@click.option("--epochs", default=20, help="Number of epochs")
def train(model: str, seq_len: int, epochs: int) -> None:
    click.echo("-"*20)
    if model == "LSTM" and seq_len == 30 and epochs == 20:
        click.echo("Training model on default parameters...")
        click.echo("-"*20)

    click.echo(f"Model type: {model}")
    click.echo(f"Sequence length: {seq_len}")
    click.echo(f"Epochs: {epochs}")
    click.echo("-"*20)
    click.echo(f"Starting Training...")



@CLI.command("evaluate", help="Evaluate the model")
@click.option("--file-name", default="best_LSTM.pt", help="File name (from outputs folder) to evaluate a given .pt file, default best_LSTM.pt")
def evaluate(file_name: str) -> None:
    click.echo("-"*20)
    click.echo(f"Evaluating {file_name}...")
    click.echo("-"*20)


@CLI.command("plot", help="Plot A Given File, if none given plot best LSTM results")
@click.option("--file-name", default="best_LSTM.pt", help="File name of the model to plot, from outputs folder")
def plot(file_name: str) -> None:
    click.echo("-"*20)
    click.echo(f"Plotting {file_name}...")
    click.echo("-"*20)




if __name__ == '__main__':
    CLI()