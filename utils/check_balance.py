import torch
from torch.utils.data import DataLoader, Subset
from utils.data_utils import EEGDataset
from progress.bar import ShadyBar
import torch.utils.data as data
from utils import params as p
import numpy as np
import torch

master = EEGDataset(p.path)  #should be test_path or something
tot_bad_channels = 0
tot_good_channels = 0

with ShadyBar(f"Computing proportions...", max=len(master)) as bar:
    for i, batch in enumerate(master):
        _, target = batch
        tot_bad_channels += np.sum(target)
        tot_good_channels += np.sum(target == 0)

        bar.next()

bp = tot_bad_channels/(len(master)*target.shape[0])
gp = tot_good_channels/(len(master)*target.shape[0])

print(f"total_badchannels: {tot_bad_channels} - total_goodchannels: {tot_good_channels}")
print("bad_channels: {} - good_channels: {} - total: {}".format(bp, gp, bp + gp))





