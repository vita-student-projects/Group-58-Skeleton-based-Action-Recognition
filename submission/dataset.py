import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

class NTU60_HRNET(Dataset):
    phase_dict = {
            'train': 'xsub_train',
            'val': 'xsub_val'
    }
    
    def __init__(self, data_path, label_path, nb_pers_max, nb_frames, nb_joints, permute_order, phase):

      super(NTU60_HRNET, self).__init__()
      print('---------------------------- Loading data ---------------------------')

      # Use GPU if available
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      # Store class attributes
      self.permute_order = permute_order
      self.phase = phase

      self.nb_pers_max = nb_pers_max
      self.nb_frames = nb_frames
      self.nb_joints = nb_joints

      self.label_dict = self.create_label_dict(label_path)
      
      # Collect the data
      self.raw_data = pickle.load(open(data_path, "rb"))
      self.data_size = np.shape(self.raw_data["split"][self.phase_dict[self.phase]])
      self.skeletons, self.labels, self.names = self.collect_data()
    

    def collect_data(self):
      skeletons = np.zeros((self.data_size[0], self.nb_pers_max, self.nb_frames, self.nb_joints, 2), dtype=np.float32)
      labels = np.zeros(self.data_size, dtype=np.int64)
      names = []

      frame_dir = []
      for i in range(len(self.raw_data['annotations'])):
        frame_dir.append(self.raw_data['annotations'][i]['frame_dir'])
      frame_dir = np.array(frame_dir)

      for i in range(self.data_size[0]):
        ind = np.argwhere(frame_dir == self.raw_data["split"][self.phase_dict[self.phase]][i]).squeeze()
        skeletons[i, :self.raw_data['annotations'][ind]['keypoint'].shape[0]] = self.raw_data['annotations'][ind]['keypoint'][:, :self.nb_frames]
        labels[i] = self.raw_data['annotations'][ind]['label']
        names.append(self.raw_data["split"][self.phase_dict[self.phase]][i])
        
      skeletons = torch.from_numpy(skeletons)
      print('before', skeletons.size())  
      skeletons = torch.permute(skeletons, self.permute_order).to(self.device)
      print('after', skeletons.size())  
      labels = torch.from_numpy(labels)
      #if self.phase == 'train':
      labels = labels.to(self.device)

      return skeletons, labels, names


    def create_label_dict(self, label_path):

      label_dict = {}
      n=0

      with open(label_path) as f:
        for line in f:
          if n == 60:
            break
          label_dict[n] = line.strip()
          n+=1

      return label_dict         
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.data_size[0]
    
    def __getitem__(self, index):
        """
        Returns the embedding, label, and image path of queried index.
        """
        skeleton = self.skeletons[index]
        label = self.labels[index]
        name = self.names[index]

        return skeleton, label, name
		
	