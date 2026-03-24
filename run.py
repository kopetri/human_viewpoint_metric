import lightning
from lightning.pytorch.cli import LightningCLI


def cli_main():
    """Entry point for LightningCLI training."""
    LightningCLI(
        lightning.LightningModule,
        lightning.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
