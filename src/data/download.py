import os
from pathlib import Path

from definitions import DATA_DIR
from definitions import LOGGER


def download_data(rewrite=False):
    path_raw_data = Path(DATA_DIR).joinpath("raw")
    path_test = path_raw_data.joinpath("PD-data-test.csv")
    path_train = path_raw_data.joinpath("PD-data-train.csv")
    path_desc = path_raw_data.joinpath("PD-data-desc.csv")

    if rewrite or not path_train.exists():
        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-train.csv -P {path_raw_data}")
        LOGGER.debug("Train dataframe successfully loaded.")
    if rewrite or not path_test.exists():
        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-test.csv -P {path_raw_data}")
        LOGGER.debug("Test dataframe successfully loaded.")
    if rewrite or not path_desc.exists():
        os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-desc.csv -P {path_raw_data}")
        LOGGER.debug("Description dataframe successfully loaded.")


if __name__ == '__main__':
    download_data(rewrite=True)
