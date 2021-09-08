from functools import partial
import pytorch_lightning as pl
from _modules.cleanEEGNet import cleanEEGNet
import _modules.params as p
from _modules.datamodule import EEGDataModule, EEGDataset
import torch
from scipy.stats import zscore
import numpy as np
import torchmetrics
import matplotlib.pyplot as plt

test_path = "/data/disk0/volkan/mbrugnara/test_08OL/"


light_mod = cleanEEGNet().load_from_checkpoint("/home/mbrugnara/cleanEEGNet/best model/models-epoch=15-valid_loss=0.00-v1.ckpt").to(p.device)
 
print(light_mod.model)

f1_comp = torchmetrics.classification.f_beta.F1()

light_mod.eval()
light_mod.freeze()
f1 = []
tests = EEGDataset(p.test_path)

lin_range = np.arange(0, 1, 0.05)

#mu = 0.3727
mu = 0.75

for test in tests:
        x, label = test
        output = torch.zeros(x.shape[1]).to(p.device) 
        shadow = torch.zeros(x.shape[1]).to(p.device)
        t_mu = torch.from_numpy(np.asarray(mu)).to(p.device)
        for i_e, epoch in enumerate(x):
            input  = torch.from_numpy(zscore(epoch.cpu()))
            input = input.to(p.device)
            partial_output = light_mod(input.view(1,1,input.shape[0],input.shape[1]))
            output = (t_mu * partial_output + (1 - t_mu) * output)
        
        round_output = torch.round(output)
        temp_f1 = f1_comp(round_output[0].cpu(), label[:,0].int())
        print(temp_f1)