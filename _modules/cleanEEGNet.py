from pytorch_lightning import LightningModule
from CNN_chocolate import ConvNet
from loss import custom_loss
import torchmetrics
import params as p
import torch


class cleanEEGNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.f1 = torchmetrics.F1()
        self.model = ConvNet()

    def forward(self, x):
        return self.model(x)
    
    def loss_fn(self, y_hat, y_target):
        loss = custom_loss()
        return loss(y_hat, y_target)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=p.lr,
                                    weight_decay=p.weigth_decay)

        return optimizer
    
    def _step(self, batch) -> Tuple[Tensor, Tensor]:
        x, label = batch
        output = self(x.float())
        
        loss = self.loss_fn(output, label.int())

        pred = torch.sigmoid(output)
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())
        
        return loss, f1

    def training_step(self, batch, batch_idx):
        loss, f1 = self._step(batch)
        
        self.log('train_loss', loss)
        self.log('train_f1', f1, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, f1 = self._step(batch)

        self.log('val_loss', loss)
        self.log('val_f1', f1)

        return loss, f1

    def test_step(self, batch, batch_idx):
        loss, f1 = self._step(batch)

        self.log('test_loss', loss)
        self.log('test_f1', f1)

        return loss, f1