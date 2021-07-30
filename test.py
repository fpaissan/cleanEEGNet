from models.cleanEEGNet import ConvNet
from torch.utils.data import DataLoader
from utils.data_utils import EEGDataset
import utils.trainer as trainer
from utils import params as p
import torch


print("Loading data...")
test_set = EEGDataset(p.test_path)
test_loader = DataLoader(test_set, batch_size=p.batch_size, shuffle=True, num_workers=p.n_workers)
print("Data loaded")

print("Loading model from checkpoint...")
model = ConvNet()
model.load_state_dict(torch.load("ckp/vague_river.ckp")['model_state_dict'])
model = model.to(p.device)

if __name__ == '__main__':
    test_loss, test_bACC, test_F1 = trainer.test(model, test_loader, None, None)

    print("Model performance on test-set: {0} f1-score, {1} bAcc".format(test_F1, test_bACC))
