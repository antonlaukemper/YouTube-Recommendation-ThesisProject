import csv
import json
import os

import pandas as pd


def read_channels(path):
    with open(path, 'rb') as f:  # opening file in binary(rb) mode
        channelData = json.load(f)
    return channelData


def write_to_file(data: pd.DataFrame, filename: str):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    data.to_csv(filename, index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="",
                escapechar=" ")

def write_df(data, content, labeled, binary):
    labeled_path = "" if labeled else "_unlabeled"
    ml_path = "binary" if binary else "multilabel"
    path = f"data_pickles/{content}{labeled_path}_{ml_path}.pl"
    data.to_pickle(path)

def load_df(content, labeled=True, binary=True):
    labeled_path = "" if labeled else "_unlabeled"
    ml_path = "binary" if binary else "multilabel"
    path = f"data_pickles/{content}{labeled_path}_{ml_path}.pl"
    return pd.read_pickle(path)
