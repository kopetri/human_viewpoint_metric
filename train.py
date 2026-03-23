from lightning.pytorch.cli import LightningCLI

from learning.dataset import ViewDataModule
from learning.model import ViewQualityModel


def cli_main():
    """Entry point for LightningCLI training."""
    LightningCLI(ViewQualityModel, ViewDataModule)


if __name__ == "__main__":
    cli_main()
