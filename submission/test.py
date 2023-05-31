import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import time
import sys

from dataset import NTU60_HRNET
from model import Model

from pyskl.models.cnns.c3d import C3D

NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2
NUM_JOINTS = 1
HEIGHT = 56
WIDTH = 56

batch_size = 50

backbone=dict(
        type='C3D',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        temporal_downsample=False)

model = C3D(in_channels=1,
        base_channels=2,
        num_stages=3,
        temporal_downsample=False)

input_size = (batch_size, NUM_JOINTS, NUM_FRAMES_MIN, HEIGHT, WIDTH)

summary(model, input_size=input_size)