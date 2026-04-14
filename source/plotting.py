import matplotlib.pyplot as plt
import click
import data
import random


def exploration_plots(typ, fp):
    # We need to get the following for plots: validation_basins, training_basins, target_variable
    ds, bi, ba = data.load_basin_data(fp)

    # print(bi)
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
    plt.show(block=True)
    return

