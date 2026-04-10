import minicamels as mc
import xarray as xr

def load_basin_data(file_path):
    dt = mc.MiniCamels(file_path)

    basin_indexes = dt.basins()

    basin_attrs = dt.attributes()

    ds = dt.load_all()

    return ds, basin_indexes, basin_attrs


def data_structure(file_path):
    ds, bi, ba = load_basin_data(file_path)
    print("Total Dataset Structure:")
    print(f"Number of Basins: {bi.shape[0]}")
    print(f"Time Span of Dataset: {str(ds['time'].to_numpy()[0])[:-19]} to {str(ds['time'].to_numpy()[-1])[:-19]}")
    var = [v for v in ds.data_vars]
    print(f"Dynamic Input Variables: {', '.join(var[:-1])}")
    print(f"Target Variable: {var[-1]}")
    print(f"Number of Static Attributes: {ba.shape[1]}")

def training_split(file_path, justif: bool = False):
    ds, bi, ba = load_basin_data(file_path)
    train_split = int(0.8 * len(bi))
    training_basins = bi[:train_split]
    validation_basins = bi[train_split:]
    if justif:
        print("Justification for Training Split:")
        print("The split schem is based on basins only. Rather than trying to do a dual\n"
              "split technique using both time and basins I opted to do the simpler route\n"
              "of just splitting by basins. This makes sure that no leakage occurs and still\n"
              "allows for sequence lengths to be determined by the user in the CLI for model\n"
              "training.")



