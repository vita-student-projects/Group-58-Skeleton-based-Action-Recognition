# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = C3dModified().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.005, betas= (0.9, 0.999), eps= 1e-08, weight_decay= 0.0, amsgrad= False)
        self.loss = nn.MSELoss()
        self.output = None

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        # pass

    def train(self, train_input, train_target, num_epochs) -> None:
        torch.set_grad_enabled(True)
        # :train_input: tensor of size (N, J, F, M, D)

        # :train_target: tensor of size (N, )
        batch_size = 50
        train_input = train_input.float().to(self.device)
        train_target = train_target.float().to(self.device)
        for epoch in range(num_epochs):
            for b in range(0, train_input.size(0), batch_size):
                output = self.model(train_input.narrow(0, b, batch_size))
                loss = self.loss(output, train_target.narrow(0, b, batch_size))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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

    def accuracy(outputs, labels):
        """
        Computes the accuracy of predictions based on the model outputs (NxK: N samples, K classes) 
        and the labels (N: N samples).
        """
        predictions = np.argmax(outputs.detach().numpy(), axis=1)
        return np.sum(predictions == labels.numpy())/len(outputs)

class C3dModified(nn.Module):
    """C3D backbone, without flatten and mlp.

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

#    def init_weights(self):
#        """Initiate the parameters either from existing checkpoint or from
#        scratch."""
#        for m in self.modules():
#            if isinstance(m, nn.Conv3d):
#                kaiming_init(m)
#        if isinstance(self.pretrained, str):
#            logger = get_root_logger()
#            logger.info(f'load model from: {self.pretrained}')
#            self.pretrained = cache_checkpoint(self.pretrained)
#            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
#
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

        return x




