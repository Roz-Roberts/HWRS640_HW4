import minicamels as mc
import xarray as xr
import click

def var_names(ds):
    var_names = [v for v in ds.data_vars]
    input_vars = var_names[:-1]
    target_var = var_names[-1]
    return input_vars, target_var


def load_basin_data(file_path):
    dt = mc.MiniCamels(file_path)

    basin_indexes = dt.basins()

    basin_attrs = dt.attributes()

    ds = dt.load_all()

    return ds, basin_indexes, basin_attrs


def load_training_basins(filepath):
    ds, bi, ba = load_basin_data(filepath)
    train_split = int(0.8 * len(bi))
    # print(ds)
    training_basins = ds.sel(basin=bi[:train_split]['basin_id'].tolist())
    validation_basins = ds.sel(basin=bi[train_split:]['basin_id'].tolist())
    # print(training_basins)
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


def preprocess(training_basins, validation_basins):
    # Training set normalization
    tb = training_basins.copy()
    tb = tb.ffill("time").bfill("time")

    varin, targ = var_names(training_basins)

    mean = tb[varin].mean(dim=("basin", "time"))
    std = tb[varin].std(dim=("basin", "time"))

    tb_norm = tb.copy()
    tb_norm[varin] = (tb[varin] - mean) / std


    # Validation set normalization
    vb = validation_basins.copy()
    vb = vb.ffill("time").bfill("time")

    mean = vb[varin].mean(dim=("basin", "time"))
    std = vb[varin].std(dim=("basin", "time"))

    vb_norm = vb.copy()
    vb_norm[varin] = (vb[varin] - mean) / std


    # Only two preprocessing steps done, first we do forward and backward fill for the dataset
    # to ensure that there are all empty values are filled in. Then we do a normalization of
    # all the input variables, as LSTM is very sensitive to variable scaling

    return tb_norm, vb_norm
