#based on: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        embedding, target = sample['embedding'], sample['target_class']
#         embedding, target = sample['embedding'], sample['target_class']
        return {'embedding': torch.from_numpy(embedding),
                'target_class': torch.from_numpy(target)}


class Emb_KGbench_Dataset(Dataset):
    """Dataset containing the embedding and the class 'hot' or 'cold' based on the temperature at that time."""

    def __init__(self, csv_file, train=True, transform=None, train_test_split=0.8, train_all=False, emb_header="emb", target_header='y'):
        csv_file = pd.read_csv(csv_file, sep=",")
        emb_classes = []
        if train_all:
            emb_classes = csv_file
        elif train:
            emb_classes = csv_file[:int(len(csv_file)*train_test_split)].reset_index()
        else:
            emb_classes = csv_file[int(len(csv_file)*train_test_split):].reset_index()

        self.embeddings = emb_classes[emb_header]
        self.targets = emb_classes[target_header]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = self.targets.iloc[idx]
        # print(target)
        target = np.array(target)
        
        sample = {'embedding': np.array([float(x.strip(' []')) for x in self.embeddings[idx].split(',')]), 
                  'target_class': target 
                 }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_n_classes(self):
        print("creating a classifier for", len(list(set(self.targets))), "classes.")
        return(len(list(set(self.targets))))

    def get_embedding_size(self):
        return len(self.embeddings[0].split(','))


class Features_KGbench_Dataset(Dataset):
    def __init__(self, csv_file, list_of_value_headers, train=True, transform=None, train_test_split=0.8, train_all=False, target_header='y'):
        csv_file = pd.read_csv(csv_file, sep=",")
        emb_classes = []
        if train_all:
            emb_classes = csv_file
        elif train:
            emb_classes = csv_file[:int(len(csv_file)*train_test_split)].reset_index()
        else:
            emb_classes = csv_file[int(len(csv_file)*train_test_split):].reset_index()

        self.features = emb_classes[list_of_value_headers].fillna(0)
        self.features = (self.features-self.features.min())/(self.features.max()-self.features.min()) #normalize values columnwise
        self.targets = emb_classes[target_header]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx:", idx)
        classes = ['hot', 'cold']
        target = self.targets.iloc[idx]
        target = np.array(target)
        
        
        sample = {'embedding': np.array(self.features.iloc[idx].fillna(0)), #added aditional fillna
                  'target_class': target 
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_n_classes(self):
        print("creating a classifier for", len(list(set(self.targets))), "classes.")
        return(len(list(set(self.targets))))