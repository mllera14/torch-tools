import os
import pandas as pd
import json

# HDF5

def h5store(filename, df, df_name, meta):
    store = pd.HDFStore(filename)
    store.put(df_name, df)
    store.get_storer(df_name).attrs.metadata = meta
    store.close()

def h5load(store_path, df_name):
    with pd.HDFStore(store_path) as store:
        data = store[df_name]
        metadata = store.get_storer(df_name).attrs.metadata

    return data, metadata
