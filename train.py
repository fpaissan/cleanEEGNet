from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import dataset_split
from models.cleanEEGNet import ConvNet
from torch.utils.data import DataLoader
from utils.data_utils import EEGDataset
import torch.utils.data as data
import utils.trainer as trainer
from utils import params as p
import torch.optim as optim
import torch
import wandb

master = EEGDataset(p.path)
train_set, val_set = dataset_split(master,p.train_val_ratio)
if p.debug:
    train_overfit = Subset(train_set, [0, 1])
    train_loader = DataLoader(train_overfit, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers)
    val_overft = Subset(val_set, [0])
    val_loader = DataLoader(val_overft, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers)
else:
    train_loader = DataLoader(train_set, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers)
    val_loader = DataLoader(val_set, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers)

model = ConvNet().to(p.device)
optimizer = optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weigth_decay)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.15, threshold=0.015, patience = 15)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

if __name__ == '__main__':

    if not p.debug:
        wandb.init(project='Bad Channel Detection')
        wandb.watch(model)

    for epoch in range(p.n_epochs):
        tb_writer = SummaryWriter(p.log_dir)
        train_loss, train_bACC, train_F1 = trainer.train(model, train_loader, optimizer, epoch, tb_writer)

        if not p.debug:
            test_loss, test_bACC, test_F1 = trainer.test(model, val_loader, scheduler, epoch, tb_writer)
        
        if not p.debug:
            wandb.log({"train loss": train_loss})
            wandb.log({"train balanced accuracy": train_bACC})
            wandb.log({"train f1 score": train_F1})
            wandb.log({"test loss": test_loss})
            wandb.log({"test balanced accuracy": test_bACC})
            wandb.log({"test f1 score": test_F1})
            wandb.log({"learning rate": [ group['lr'] for group in optimizer.param_groups ][-1]})

            if epoch == 0:
                min_loss = test_loss
            if (min_loss > test_loss):
                print("\nSaving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                }, p.ckp_dir + str(epoch) + "_min_loss.ckp")
                min_loss = test_loss

