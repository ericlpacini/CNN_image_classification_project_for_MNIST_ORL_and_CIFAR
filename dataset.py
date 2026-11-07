import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class CustomImageDataset(Dataset):
    def __init__(self, dataset_name, split='train'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, dataset_name)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.instances = data[split]
        self.dataset_name = dataset_name.upper()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        img = np.array(instance['image'], dtype=np.float32)
        label = int(instance['label'])

        if self.dataset_name == 'MNIST':
            img = torch.tensor(img).unsqueeze(0) / 255.0
            label = torch.tensor(label, dtype=torch.long)

        elif self.dataset_name == 'ORL':
            img = img[:, :, 0]
            img = torch.tensor(img).unsqueeze(0) / 255.0
            label = torch.tensor(label - 1, dtype=torch.long)

        elif self.dataset_name == 'CIFAR':
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img) / 255.0
            label = torch.tensor(label, dtype=torch.long)

        return img, label