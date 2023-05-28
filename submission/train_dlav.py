import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from torchinfo import summary
import pickle
import numpy as np
from c3d_modified_dlav import C3dModified, Model
import os
import time 

NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2

X_train = torch.zeros(4000, 17, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D)
y_train = torch.zeros(4000)
X_test = torch.zeros(16000, 17, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D)
y_test = torch.zeros(16000)

print('What we want', X_train.shape, y_train.shape)

# --------------------- Load data ---------------------------------
"""
data = pickle.load(open('../../../data/nturgbd/ntu60_hrnet.pkl', "rb"))

skeletons = []
labels = []
frame_dir = []

for i in range(len(data['annotations'])):
  skeletons.append(data['annotations'][i]['keypoint'])
  labels.append(data['annotations'][i]['label'])
  frame_dir.append(data['annotations'][i]['frame_dir'])

frame_dir = np.array(frame_dir)

X_train = np.zeros((np.shape(data["split"]['xsub_train'])[0], NUM_PERS_MAX, NUM_FRAMES_MIN, 17, 2))
y_train = np.zeros(np.shape(data["split"]['xsub_train']))

for i in range(np.shape(data["split"]['xsub_train'])[0]):
  ind = np.argwhere(frame_dir == data["split"]['xsub_train'][i]).squeeze()
  temp_skull = skeletons[ind]
  temp_label = labels[ind]
  X_train[i, :temp_skull.shape[0]] = temp_skull[:, :32]
  y_train[i] = temp_label

X_val = np.zeros((np.shape(data["split"]['xsub_val'])[0], NUM_PERS_MAX, NUM_FRAMES_MIN, 17, 2))
y_val = np.zeros(np.shape(data["split"]['xsub_val']))

for i in range(np.shape(data["split"]['xsub_val'])[0]):
  ind = np.argwhere(frame_dir == data["split"]['xsub_val'][i]).squeeze()
  temp_skull = skeletons[ind]
  temp_label = labels[ind]
  X_val[i, :temp_skull.shape[0]] = temp_skull[:, :32]
  y_val[i] = temp_label

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)

X_train = torch.permute(X_train, (0, 3, 2, 1, 4))
X_val = torch.permute(X_val, (0, 3, 2, 1, 4))

print('What we have', X_train.shape, y_train.shape)

"""
# Train model
num_epochs = 1
print('Hello world!')
# TestModelDlav = C3dModified()
# print(summary(C3dModified(), input_size=(5, 17, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D)))
# TestModelDlav.forward(X_train[0:5])

TestModelDlav = Model()
start = time.perf_counter()
TestModelDlav.train(X_train[0:1000], y_train[0:1000], num_epochs)
stop = time.perf_counter()
print(f'The training is done in {stop - start} seconds')
