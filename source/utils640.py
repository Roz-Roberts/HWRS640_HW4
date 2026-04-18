import click
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
    click.echo(f"Model input: previous {seq_len} days of {', '.join(data.M)}")
    click.echo(f"Target: {veris[1]} at day t + {horizon}")
    click.echo(f"Each supervised sample uses a sliding window of {seq_len} days as input\n"
        f"and predicts the target {horizon} day(s) after the end of that window.")
    return