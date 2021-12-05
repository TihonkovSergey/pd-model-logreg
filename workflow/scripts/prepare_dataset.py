import click

from src.data.dataframes import get_train, get_test
from src.features.feature_extraction import prepare_dataset


@click.command()
def prepare_dataset():
    train_df = get_train()
    test_df = get_test()
    prepare_dataset(train_df, test_df, load_from_disk=False)
