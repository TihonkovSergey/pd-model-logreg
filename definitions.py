import os
import logging
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

REPORT_DIR = ROOT_DIR.joinpath("reports")
REPORT_DIR.mkdir(exist_ok=True)

DATA_DIR = ROOT_DIR.joinpath("data")
DATA_DIR.mkdir(exist_ok=True)


SEED = 0

USE_PRECALC = True  # if True some calculations will not be performed

N_SPLITS = 5  # number of splits in cross validation
N_FEATURES_FR = 15  # number of features for data with financial report
N_FEATURES_NFR = 10  # number of features for data without financial report

THRESHOLD_TUNING_N_SAMPLES = 100
THRESHOLD_TUNING_N_ITERS = 20

TARGET_NAME = 'default_12m'

LOG_DIR = ROOT_DIR.joinpath("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR.joinpath("logs")
logging.basicConfig(filename=LOG_PATH,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger("Main_Logger")

