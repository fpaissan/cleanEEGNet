
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import LeavePGroupsOut
from pytorch_lightning import LightningDataModule
from pickle import load as p_load
from scipy.stats import zscore
from numpy import unique
from pathlib import Path
from torch import Tensor
from typing import Tuple
import numpy as np


class EEGDataset(Dataset):
    def __init__(self, 
                parent_dir: Path,
                split: str) -> None:
        self.data_dir = parent_dir.joinpath(split)
        self.files = sorted(self.data_dir.iterdir())
        self.files = unique(
            ['_'.join(str(f.name).split('_')[:-1]) for f in self.files]
        )
        print(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self,
                    i: int) -> Tuple[np.array, np.array]:
        label_path = self.data_dir.joinpath('_'.join([self.files[i]] + ['labels.pkl']))
        data_path = self.data_dir.joinpath('_'.join([self.files[i]] + ['data.pkl']))

        # Load data
        with open(data_path, "rb") as f:
            data = p_load(f)[:, 0:330000]

        # Load label
        with open(label_path, "rb") as f:
            labels = p_load(f)

        data = zscore(data)
        data = np.expand_dims(data, 0)
        return data, labels


class EEGDataModule(LightningDataModule):
    """Data Module for OpenNeuro bad channel dataset"""
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
    
    def _dataset_split(self, dataset : EEGDataset, ratio: int) \
    -> Tuple[Subset, Subset]:
        """
        Performs GroupSplit to avoid having the same subject in train and val
        """
        # Extracts sub IDs from filename
        groups = [str(f).split("_")[0] for f in dataset.files]
        # Splits based on group
        splitter = LeavePGroupsOut(n_groups=int(np.ceil(len(np.unique(groups))*ratio)))
        temp = list(splitter.split(dataset.files, groups=groups))
        train_index, test_index = temp[np.random.randint(0, splitter.get_n_splits(groups=groups))]

        train_split = Subset(dataset, train_index)
        val_split = Subset(dataset, test_index)

        return train_split, val_split

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set, self.val_set = self._dataset_split(
                EEGDataset(Path("/badchannel"), "train"), 
                ratio=0.2
            )
            
            print(len(self.train_set), len(self.val_set))
        elif stage == "test":
            self.test_set = EEGDataset(Path("/badchannel"), "test")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, num_workers=4, persistent_workers=True)
