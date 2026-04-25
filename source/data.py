import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import minicamels as mc
import click


class MiniCamelsDataset(Dataset):
    def __init__(
            self,
            ds,
            input_vars=None,
            target_var="qobs",
            seq_len=30,
            horizon=1,
            basin_ids=None,
            skip_nan=True,
            normalize_x=False,
            x_mean=None,
            x_std=None,
            normalize_y=False,
            y_mean=None,
            y_std=None,
            fill_nan=True
    ):
        """
        PyTorch Dataset for MiniCamels xarray output.

        Parameters
        ----------
        ds : xr.Dataset
            Xarray dataset with dims (basin, time)
        input_vars : list[str]
            Dynamic input variables to use. Defaults to all data_vars except target_var.
        target_var : str
            Target variable name.
        seq_len : int
            Number of past timesteps in each input sequence.
        horizon : int
            Forecast horizon. horizon=1 means predict one day ahead.
        basin_ids : list[str] or None
            Optional subset of basins to use.
        skip_nan : bool
            If True, skip samples containing NaNs in x or y.
        normalize_x : bool
            Whether to normalize inputs.
        x_mean, x_std : np.ndarray or None
            Per-feature normalization stats for inputs.
        normalize_y : bool
            Whether to normalize target.
        y_mean, y_std : float or np.ndarray or None
            Normalization stats for target.
        """
        if fill_nan:
            ds = ds.ffill(dim="time").bfill(dim="time")

        self.ds = ds

        if basin_ids is not None:
            self.ds = self.ds.sel(basin=basin_ids)

        self.target_var = target_var
        self.seq_len = seq_len
        self.horizon = horizon
        self.skip_nan = skip_nan
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        if input_vars is None:
            input_vars = [v for v in self.ds.data_vars if v != target_var]

        self.input_vars = input_vars

        self.n_features = len(self.input_vars)

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        # Create valid sample index map of (basin_id, target_time_index)

        self.samples = []
        self._build_index()

    def _build_index(self):
        basin_vals = self.ds["basin"].values
        n_time = self.ds.sizes["time"]

        for basin in basin_vals:
            basin_ds = self.ds.sel(basin=basin)

            # Get shape data in the form of (time, features)

            x_all = np.stack([basin_ds[var].to_numpy() for var in self.input_vars], axis=-1)

            y_all = basin_ds[self.target_var].to_numpy()

            # target index (t) will need an input window of (t-seq_len, t)
            # and forecast y at t+horizon-1 (-1 is to make sure we are selecting
            # the right horizon result

            for t in range(self.seq_len, n_time-self.horizon+1):
                x_window = x_all[(t-self.seq_len):t, :]
                y_target = y_all[t+self.horizon - 1]

                if self.skip_nan:
                    if np.isnan(x_window).any() or np.isnan(y_target).any():
                        continue

                self.samples.append((basin, t))

    def __len__(self):
        return len(self.samples)

    def get_var_info(self):
        return [self.input_vars, self.target_var]

    def __getitem__(self, idx):
        basin, t = self.samples[idx]
        basin_ds = self.ds.sel(basin=basin)

        x = np.stack([basin_ds[var].to_numpy()[t-self.seq_len:t] for var in self.input_vars], axis=-1)
        # x has shape (seq_len, n_features)

        y = basin_ds[self.target_var].to_numpy()[t+self.horizon-1]


        if self.normalize_x:
            if self.x_mean is None or self.x_std is None:
                raise ValueError("x_mean and x_std must be provided when normalize_x=True")
            x = (x - self.x_mean) / self.x_std

        if self.normalize_y:
            if self.y_mean is None or self.y_std is None:
                raise ValueError("y_mean and y_std must be provided when normalize_y=True")
            y = (y - self.y_mean) / self.y_std

        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32, copy=True)

        return torch.from_numpy(x), torch.from_numpy(y)


def compute_input_stats(ds, input_vars):
    """
    Compute per-variable mean/std over (basin, time) for input variables.
    Returns numpy arrays of shape (n_features,).
    """
    mean = ds[input_vars].mean(dim=("basin", "time")).to_array().to_numpy()
    std = ds[input_vars].std(dim=("basin", "time")).to_array().to_numpy()

    # Avoid divide-by-zero
    std = np.where(std == 0, 1.0, std)
    return mean, std
def compute_target_stats(ds, target_var="qobs"):
    """
    Compute mean/std for target variable over (basin, time).
    Returns scalars.
    """
    mean = ds[target_var].mean(dim=("basin", "time")).item()
    std = ds[target_var].std(dim=("basin", "time")).item()

    if std == 0:
        std = 1.0
    return mean, std
def split_dataset_by_basin(ds, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42, shuffle=True):
    """
    Splits an xarray Dataset into train/val/test subsets by basin.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimension 'basin'
    train_frac, val_frac, test_frac : float
        Fractions for the split. Must sum to 1.0
    seed : int
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle basin order before splitting

    Returns
    -------
    train_ds, val_ds, test_ds : xr.Dataset
        Dataset subsets split by basin
    train_basins, val_basins, test_basins : np.ndarray
        Arrays of basin IDs for each split
    """
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    basin_ids = ds["basin"].values.copy()
    n_basins = len(basin_ids)

    if shuffle:
        rng = np.random.default_rng(seed)
        basin_ids = rng.permutation(basin_ids)

    n_train = int(n_basins * train_frac)
    n_val = int(n_basins * val_frac)
    n_test = n_basins - n_train - n_val

    train_basins = basin_ids[:n_train]
    val_basins = basin_ids[n_train:n_train + n_val]
    test_basins = basin_ids[n_train + n_val:]

    train_ds = ds.sel(basin=train_basins)
    val_ds = ds.sel(basin=val_basins)
    test_ds = ds.sel(basin=test_basins)

    return train_ds, val_ds, test_ds, train_basins, val_basins, test_basins




def make_dataloaders(train_ds, val_ds,test_ds, input_vars=None, target_var="qobs", seq_len=30, horizon=1, batch_size=64, shuffle_train=True, normalize_x=True, normalize_y=False):
    if input_vars is None:
        input_vars = [v for v in train_ds.data_vars if v != target_var]
    x_mean, x_std = compute_input_stats(train_ds, input_vars)

    if normalize_y:
        y_mean, y_std = compute_target_stats(train_ds, target_var)
    else:
        y_mean, y_std = None, None

    train_dataset = MiniCamelsDataset(
        ds=train_ds,
        input_vars=input_vars,
        target_var=target_var,
        seq_len=seq_len,
        horizon=horizon,
        skip_nan=True,
        normalize_x=normalize_x,
        x_mean=x_mean,
        x_std=x_std,
        normalize_y=normalize_y,
        y_mean=y_mean,
        y_std=y_std
    )

    val_dataset = MiniCamelsDataset(
        ds=val_ds,
        input_vars=input_vars,
        target_var=target_var,
        seq_len=seq_len,
        horizon=horizon,
        skip_nan=True,
        normalize_x=normalize_x,
        x_mean=x_mean,
        x_std=x_std,
        normalize_y=normalize_y,
        y_mean=y_mean,
        y_std=y_std
    )

    test_dataset = MiniCamelsDataset(
        ds=test_ds,
        input_vars=input_vars,
        target_var=target_var,
        seq_len=seq_len,
        horizon=horizon,
        skip_nan=True,
        normalize_x=normalize_x,
        x_mean=x_mean,
        x_std=x_std,
        normalize_y=normalize_y,
        y_mean=y_mean,
        y_std=y_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def load_basin_data(file_path):
    if file_path is None:
        dt = mc.MiniCamels()
    else:
        dt = mc.MiniCamels(file_path)

    basin_indexes = dt.basins()

    basin_attrs = dt.attributes()

    ds = dt.load_all()

    train_ds, val_ds, test_ds, train_basins, val_basins, test_basins = split_dataset_by_basin(ds)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = make_dataloaders(train_ds, val_ds,test_ds)

    return ds, basin_indexes, basin_attrs, train_dataset.get_var_info()


def training_split_justification():
    click.echo("Justification for Training Split:")
    click.echo("The split schem is based on basins only. Rather than trying to do a dual\n"
          "split technique using both time and basins I opted to do the simpler route\n"
          "of just splitting by basins. This makes sure that no leakage occurs and still\n"
          "allows for sequence lengths to be determined by the user in the CLI for model\n"
          "training.")
    return

