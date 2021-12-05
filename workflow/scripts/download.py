import click
from src.data.download import download_data


@click.command()
@click.option("--rewrite", help="Rewrite if exists.", is_flag=True)
def download_data(rewrite):
    download_data(rewrite=rewrite)


if __name__ == '__main__':
    download_data()
