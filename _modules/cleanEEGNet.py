from pytorch_lightning import LightningModule
from _modules.CNN_chocolate import ConvNet
from loss import custom_loss
import torchmetrics
import params as p
import torch
from torch import nn
from torch.autograd import Variable
from scipy.stats import zscore
import numpy as np

class cleanEEGNet(LightningModule):
    def __init__(self):
        
        super().__init__()
        self.f1 = torchmetrics.classification.f_beta.F1()
        self.model = ConvNet()
        self.mu = nn.Parameter(-0.35*torch.ones(1).to(p.device))
        self.mu_sigmoid = torch.zeros(1)

        self.null_predictions = 0
        self.num_predictions = 0
    def forward(self, x):
        output = torch.zeros(x.shape[0],x.shape[2]).to(p.device) # (n_batches, n_channels)
        shadow = torch.zeros(x.shape[2]).to(p.device)
        self.mu_sigmoid = torch.sigmoid(self.mu).to(p.device)
        for i_b, batch in enumerate(x):
            for i_e, epoch in enumerate(batch):
                #print("std deviation: ",np.std(epoch.cpu().numpy()), " mean: ", np.mean(epoch.cpu().numpy()))
                input  = torch.from_numpy(zscore(epoch.cpu()))
                #print(input.shape)
                #print("norm std deviation: ",np.std(input.numpy()), " norm mean: ", np.mean(input.numpy()))
                input = input.to(p.device)
                output[i_b,:] = (self.mu_sigmoid * self.model.forward(input.view(1,1,input.shape[0],input.shape[1])) + (1 - self.mu_sigmoid) * shadow)
                shadow = output[i_b,:].clone()
        return output
    
    def loss_fn(self, y_hat, y_target):
        loss = custom_loss()
        return loss(y_hat, y_target)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=p.lr,
                                    weight_decay=p.weigth_decay)


        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.7)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            cooldown=5,
            min_lr=1e-8,
        )'''

        return {"optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    

    def training_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())       
        loss = self.loss_fn(output, label.int())
        #print("output: ", torch.sigmoid(output), "labels: ", label)
        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('mu ',self.mu_sigmoid)

        
        '''self.num_predictions += 1
        if(torch.sum(pred) == 0):
            self.null_predictions+=1

        print(int(self.null_predictions) , " / ", int(self.num_predictions), " predictions are all 0s")'''

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())

        loss = self.loss_fn(output, label.int())

        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('val_loss', loss)
        self.log('val_f1', f1)

        return loss, f1

    def test_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())

        loss = self.loss_fn(output, label.int())

        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('test_loss', loss)
        self.log('test_f1', f1)

        return loss, f1