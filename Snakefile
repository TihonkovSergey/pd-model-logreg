from pathlib import Path

rule all:
    input:
        Path("reports/submit.csv")

rule download:
    output:
        Path("data/raw/PD-data-desc.csv"),
        Path("data/raw/PD-data-test.csv"),
        Path("data/raw/PD-data-train.csv")
    script:
        "workflow/scripts/download.py"

rule preprocessing:
    input:
        Path("data/raw/PD-data-test.csv"),
        Path("data/raw/PD-data-train.csv")
    output:
        Path("data/prepared/prepared_test.csv"),
        Path("data/prepared/prepared_train_with_financial_report.csv"),
        Path("data/prepared/prepared_train_without_financial_report.csv"),
    script:
        "workflow/scripts/prepare_dataset.py"

rule submission:
    input:
        Path("data/prepared/prepared_test.csv"),
        Path("data/prepared/prepared_train_with_financial_report.csv"),
        Path("data/prepared/prepared_train_without_financial_report.csv"),
    output:
        Path("reports/submit.csv")
    script:
        "workflow/scripts/make_submission.py"
