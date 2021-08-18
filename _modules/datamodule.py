
import pickle
import torch
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import LeavePGroupsOut
from pytorch_lightning import LightningDataModule
from pickle import load as p_load
from scipy.stats import zscore
from pathlib import Path
from torch import Tensor
from typing import Tuple
import numpy as np
import params as p


class EEGDataset(Dataset):
    def __init__(self, 
                parent_dir: Path,
                split: str) -> None:
        self.data_dir = parent_dir.joinpath(split)
        self.files = sorted(self.data_dir.iterdir())

    def __init__(self, dir) -> None:
        self.data_dir = Path(dir)
        self.files = sorted(self.data_dir.iterdir())

    def __len__(self) -> int:
        return int(len(self.files)/2)

    def __getitem__(self, i: int):
        print(i)
        fname = str(self.files[i].name).split('_')[:-1]
        label_path = self.files[i].parents[0].joinpath('_'.join(fname + ['labels.pkl']))
        data_path = self.files[i].parents[0].joinpath('_'.join(fname + ['data.pkl']))

        # Load data
        with open(data_path, "rb") as f:
            data = p_load(f)[:, 0:330000]
    
        windowSampleNum = p.sampleRate * p.windowLength
        windowedData = torch.FloatTensor(data[:,0:windowSampleNum]).unsqueeze(0)
        slidingIndex = int(windowSampleNum * p.overlap)

        while(slidingIndex < 330000 - windowSampleNum):
            windowedData  = torch.cat((windowedData, (torch.FloatTensor(data[:,slidingIndex:slidingIndex+windowSampleNum]).unsqueeze(0))),0)
            slidingIndex += int(windowSampleNum * p.overlap)
        
        # Load label
        with open(label_path, "rb") as f:
            ch_labels = p_load(f)

        labels = torch.ones((windowedData.shape[0],windowedData.shape[1]))
        for i in range(ch_labels.shape[0]) :
            labels[i,:] *= ch_labels[i]
        
        data = zscore(windowedData)
        data = windowedData.unsqueeze(0)
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
        groups = [str(f.name).split("_")[0] for f in dataset.files]
        # Splits based on group
        splitter = LeavePGroupsOut(n_groups=int(np.ceil(len(np.unique(groups))*ratio)))
        temp = list(splitter.split(dataset.files, groups=groups))
        train_index, test_index = temp[np.random.randint(0, splitter.get_n_splits(groups=groups))]

        train_split = Subset(dataset, train_index)
        val_split = Subset(dataset, test_index)

        return train_split, val_split

    def setup(self, stage=None):
        self.train_set, self.val_set = self._dataset_split(
            EEGDataset(p.path), 
            ratio=0.8
            )

        self.test_set = EEGDataset(p.test_path)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, self.batch_size, num_workers=4, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, num_workers=4, persistent_workers=True)

    def save_files(self):
        root = "./badchannels/"
        #train set
        for i in range(len(self.train_set)):
            print((self.train_set[i])[0].shape)



