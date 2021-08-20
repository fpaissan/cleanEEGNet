from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from CNN_chocolate import ConvNet

from _modules.datamodule import EEGDataModule, EEGDataset
from _modules.cleanEEGNet import cleanEEGNet

import params as p
import os

mod = cleanEEGNet()
data_module = EEGDataModule(2)

os.environ['WANDB_MODE'] = "online" 

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
                    num_sanity_val_steps=0,
                    logger=wandb_logger

        )



trainer.fit(model=mod, datamodule=data_module)
trainer.test(datamodule=data_module, verbose=1)
