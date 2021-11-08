from pathlib import Path

import pandas as pd

from definitions import ROOT_DIR


def get_train():
    data_path = Path(ROOT_DIR).joinpath("data/raw/")
    return pd.read_csv(data_path.joinpath('PD-data-train.csv'), sep=';')


def get_test():
    data_path = Path(ROOT_DIR).joinpath("data/raw/")
    return pd.read_csv(data_path.joinpath('PD-data-test.csv'), sep=';')


def get_data_description():
    data_path = Path(ROOT_DIR).joinpath("data/raw/")
    return pd.read_csv(data_path.joinpath('PD-data-desc.csv'), sep=';')
