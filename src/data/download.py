import os
from definitions import ROOT_DIR
from pathlib import Path


def download_data():
    data_path = Path(ROOT_DIR).joinpath("data/raw/")
    os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-train.csv -P {data_path}")
    os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-test.csv -P {data_path}")
    os.system(f"wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-desc.csv -P {data_path}")


if __name__ == '__main__':
    download_data()
