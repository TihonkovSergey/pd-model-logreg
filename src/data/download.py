import os

if __name__ == '__main__':
    os.system("wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-train.csv -P data/raw/")
    os.system("wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-test.csv -P data/raw/")
    os.system("wget https://raw.githubusercontent.com/BKHV/risk_models/master/data/PD-data-desc.csv -P data/raw/")
