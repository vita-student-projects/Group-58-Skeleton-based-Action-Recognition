import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from torchinfo import summary
import pickle
import numpy as np
from c3d_modified_dlav import C3dModified, Model
import os
import time

from dataset import NTU60_HRNET

NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2
NUM_JOINTS = 17

batch_size = 50
num_epochs = 25

data_path = f'{os.getcwd()}/data/nturgbd/ntu60_hrnet.pkl'
label_map_path = f'{os.getcwd()}/tools/data/label_map/nturgbd_120.txt'


train_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, "train")
val_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, "val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Train model

print('Hello world!')

print(summary(C3dModified(), input_size=(batch_size, NUM_JOINTS, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D)))

TestModelDlav = Model()
start = time.perf_counter()
TestModelDlav.training(train_loader, val_loader, num_epochs, f'{os.getcwd()}/outputs/')
stop = time.perf_counter()
print(f'The training is done in {stop - start} seconds')
