import torch

path = "/data/disk0/volkan/mbrugnara/train"
test_path = "/data/disk0/volkan/mbrugnara/test"

debug = False

log_dir = "tb_log/"
ckp_dir = "ckp/"
k_w = 512  # Kernel width for model CNN_vanilla.py
s_w = 256  # Stride width fot model CNN_vanilla.py

ch_out1 = 32  # Output channels for CNN_chocolate.py

# Parameters for paprika model
ch_out1p = 64
ch_out2p = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
batch_size = 2
n_workers = 8

n_epochs = 100000

train_val_ratio = 0.8

dataset_min_length = 330000

tot_bad_channels = 348
tot_good_channels = 5480

pos_weight = tot_bad_channels / tot_good_channels
#pos_weight = 1/(tot_bad_channels/(tot_good_channels+tot_bad_channels))/2

dp_rate = 0.1 # Dropout rate
weigth_decay = 1e-2  # Weight decay factor on Adam
lr = 1e-2

sampleRate = 512
overlap = 0.5
windowLength = 5 #in seconds
