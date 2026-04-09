import minicamels as mc

def load_basin_data():
    global ds
    dt = mc.MiniCamels("C:\\Users\\roswe\\Desktop\\minicamels\\data")

    basin_indexes = dt.basins()

    basin_attrs = dt.attributes()

    ds = dt.load_all()

    return ds, basin_indexes, basin_attrs


def data_structure():
    _, bi, ba = load_basin_data()
    print(ds)


data_structure()
