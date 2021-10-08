from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from _modules.datamodule import EEGDataModule
from _modules.cleanEEGNet import cleanEEGNet

if __name__ == "__main__":
        mod = cleanEEGNet()
        data_module = EEGDataModule()

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                dirpath='./ckp',
                filename='models-{epoch:02d}-{valid_loss:.2f}',
                save_top_k=3,
                mode='min')

        wandb_logger = WandbLogger()
        trainer = pl.Trainer(gpus=1,
                        max_epochs=150,
                        callbacks=[checkpoint_callback],
                        logger=wandb_logger
                )

        trainer.fit(model=mod, datamodule=data_module)
        trainer.test(datamodule=data_module, verbose=1)
