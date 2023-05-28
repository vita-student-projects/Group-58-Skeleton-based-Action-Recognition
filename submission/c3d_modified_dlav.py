# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from torchinfo import summary
import pickle
import numpy as np
from pathlib import Path


#from ...utils import cache_checkpoint, get_root_logger
#from ..builder import BACKBONES

# Datashape num_joints x num_frames x 2D coord. x num_pers
NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2

# -------------------- Model --------------------------------

torch.set_grad_enabled(True)
class Model():
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print('device', self.device)
        self.model = C3dModified().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.005)
        self.criterion = nn.MSELoss()
        self.output = None

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

    def train(self, loader):
        # Set the model in train mode
        self.model.training = True

        # Iterate over the batches
        full_outputs = []
        full_labels = []
        losses = []
        for batch in loader:
            skeletons, labels, _ = batch
            labels_one_hot = F.one_hot(labels, 60).float()
            pred = self.model(skeletons)
            loss = self.criterion(pred, labels_one_hot)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            full_outputs.append(pred)
            full_labels.append(labels)
            losses.append(loss)

        # Concat
        full_outputs = torch.cat(full_outputs).cpu()
        full_labels = torch.cat(full_labels).cpu()
        losses = torch.stack(losses).mean().cpu()

        acc = self.accuracy(full_outputs, full_labels)
        return acc, full_outputs, full_labels, losses
		
    @torch.no_grad()
    def validate(self, loader):
        self.model.training = True

        full_outputs = []
        full_labels = []
        losses = []
        for batch in loader:
            skeletons, labels, _ = batch
            labels_one_hot = F.one_hot(labels, 60).float()
            pred = self.model(skeletons)
            loss = self.criterion(pred, labels_one_hot)
            full_outputs.append(pred)
            full_labels.append(labels)
            losses.append(loss)

        full_outputs = torch.cat(full_outputs).cpu()
        full_labels = torch.cat(full_labels).cpu()
        losses = torch.stack(losses).mean().cpu()

        acc = self.accuracy(full_outputs, full_labels)
        return acc, full_outputs, full_labels, losses
	
    def training(self, train_loader, val_loader, nb_epochs, model_savepath):
        epochs = nb_epochs
        best_acc = 0

        list_train_acc = []
        list_train_loss = []
        list_val_acc = []
        list_val_loss = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # Train
            train_acc, _, _, train_loss = self.train(train_loader)
            print(f"training metrics : accuracy = {train_acc}  loss = {train_loss}")
            list_train_acc.append(train_acc)
            list_train_loss.append(train_loss)

            # Evaluate
            val_acc, _, _, val_loss = self.validate(val_loader)
            print(f"validation metrics : accuracy = {val_acc}  loss = {val_loss}")
            list_val_acc.append(val_acc)
            list_val_loss.append(val_loss)

            # Save the model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model, model_savepath + 'bestmodel.pth')

        np.save(model_savepath+'train_acc.npy', list_train_acc)
        np.save(model_savepath+'train_loss.npy', list_train_loss)
        np.save(model_savepath+'val_acc.npy', list_val_acc)
        np.save(model_savepath+'val_loss.npy', list_val_loss)

        fig, axs = plt.subplots(2,1, figsize=(20,10))

        axs[0].plot(list_train_acc, label='train')
        axs[0].plot(list_val_acc, label='validation')
        axs[0].legend()
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Accuracy")

        axs[1].plot(list_train_loss, label='train')
        axs[1].plot(list_val_loss, label='validation')
        axs[1].legend()
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Loss")

        plt.savefig(model_savepath+ "fig_train_skeleton.png")
        plt.show()




    def save_model(self):
        torch.save(self.model.state_dict(), 'bestmodel.pth')

    def predict(self, test_input) -> torch.Tensor:
        # : test_input : tensor of size (N1, C, H, W) that has to be denoised by the trained
        # or the loaded network.

        # : returns a tensor of the size (N1, C, H, W)
        test_input = test_input.float().to(self.device)
        self.output = self.model(test_input)
        self.output = torch.where(self.output <= 255, self.output, torch.tensor([255]).type(torch.FloatTensor).to(self.device))
        self.output = self.output.cpu()
        return self.output

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of predictions based on the model outputs (NxK: N samples, K classes) 
        and the labels (N: N samples).
        """
        predictions = np.argmax(outputs.detach().numpy(), axis=1)
        return np.sum(predictions == labels.numpy())/len(outputs)

class C3dModified(nn.Module):
    """C3D backbone, with flatten and without mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=17,
                 base_channels=64,
                 num_stages=4,
                 temporal_downsample=True,
                 pretrained=None):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample
        pool_kernel, pool_stride = (3, 2, 2), (2, 1, 1)
        if not self.temporal_downsample:
            print('hey brUAW')
            pool_kernel, pool_stride = (3, 2, 2), (2, 1, 1)

        c3d_conv_param = dict(kernel_size=(3, 2, 2), padding=(1, 1, 1), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        #print(self.conv1a.weight.dtype)
        self.pool1 = nn.AvgPool3d(kernel_size=(3, 2, 2), stride=(2, 1, 1))

        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
        
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, 60)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """
        x = self.conv1a(x)

        x = self.pool1(x)
        x = self.conv2a(x)

        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        if self.num_stages == 3:
            return x

        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        return x




