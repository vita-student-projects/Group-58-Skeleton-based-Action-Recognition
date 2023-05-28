import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from torchinfo import summary
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# Datashape num_joints x num_frames x 2D coord. x num_pers
NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2


torch.set_grad_enabled(True)
class Model():
    def __init__(self, model) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        print('device', self.device)
        self.model = model.to(self.device)

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
        for batch in tqdm(loader):
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
        for batch in tqdm(loader):
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
        # Check if the folder exists
        if not os.path.exists(model_savepath):
          # If the folder does not exist, create it
          os.makedirs(model_savepath)

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
            list_train_loss.append(train_loss.detach().numpy())

            # Evaluate
            val_acc, _, _, val_loss = self.validate(val_loader)
            print(f"validation metrics : accuracy = {val_acc}  loss = {val_loss}")
            list_val_acc.append(val_acc)
            list_val_loss.append(val_loss.detach().numpy())

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
        axs[0].set_title('Accuracy')
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Accuracy")

        axs[1].plot(list_train_loss, label='train')
        axs[1].plot(list_val_loss, label='validation')
        axs[1].legend()
        axs[1].set_title('Loss')
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("MSE")

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