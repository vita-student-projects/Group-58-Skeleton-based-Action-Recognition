import torch
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import time

from dataset import NTU60_HRNET
from train import Model
from c3d_modified_dlav import C3dModifiedJointFramePersDim, C3dModifiedPersJointFrameDim

torch.cuda.empty_cache()

NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2
NUM_JOINTS = 17

batch_size = 1
num_epochs = 2

data_path = f'{os.getcwd()}/data/nturgbd/ntu60_hrnet.pkl'
label_map_path = f'{os.getcwd()}/tools/data/label_map/nturgbd_120.txt'
video_path = f'{os.getcwd()}/ntu_samples/'

model_name = 'pjfd'
model_savepath = f'{os.getcwd()}/outputs/{model_name}/'

if model_name == 'jfpd':
  permute_order = (0, 3, 2, 1, 4)
  model = C3dModifiedJointFramePersDim()
  input_size = (batch_size, NUM_JOINTS, NUM_FRAMES_MIN, NUM_PERS_MAX, COORD_2D) 
elif model_name == 'pjfd':
  permute_order = (0, 1, 2, 3, 4)
  model = C3dModifiedPersJointFrameDim()
  input_size = (batch_size, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, COORD_2D)


test_dataset = NTU60_HRNET(data_path, label_map_path, NUM_PERS_MAX, NUM_FRAMES_MIN, NUM_JOINTS, permute_order, "val")
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

summary(model, input_size=input_size)

TestModelDlav = Model(model)
TestModelDlav.load_pretrained_model(model_savepath +  "bestmodel.pth")
start = time.perf_counter()
acc, full_outputs, full_labels, loss, full_names = TestModelDlav.predict(test_loader)
stop = time.perf_counter()
print(f'The prediction is done in {stop - start} seconds for {len(test_dataset)} predictions.')
print(f"testing metrics : accuracy = {acc}  loss = {loss}")

TestModelDLav.show_prediction(test_dataset.skeletons[0].numpy(), full_outputs[0], full_labels[0], full_names[0], test_dataset.label_dict, model_savepath, video_path)
