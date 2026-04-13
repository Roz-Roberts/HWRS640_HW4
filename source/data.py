import minicamels as mc
import xarray as xr
import click

def load_basin_data(file_path):
    dt = mc.MiniCamels(file_path)

    basin_indexes = dt.basins()

    basin_attrs = dt.attributes()

    ds = dt.load_all()

    return ds, basin_indexes, basin_attrs


def load_training_basins(filepath):
    _, bi, ba = load_basin_data(filepath)
    train_split = int(0.8 * len(bi))
    training_basins = bi[:train_split]
    validation_basins = bi[train_split:]
    return training_basins, validation_basins


def training_split(file_path, justif: bool = False):
    training_basins, validation_basins = load_training_basins(file_path)
    if justif:
        click.echo("Justification for Training Split:")
        click.echo("The split schem is based on basins only. Rather than trying to do a dual\n"
              "split technique using both time and basins I opted to do the simpler route\n"
              "of just splitting by basins. This makes sure that no leakage occurs and still\n"
              "allows for sequence lengths to be determined by the user in the CLI for model\n"
              "training.")
    return training_basins, validation_basins

