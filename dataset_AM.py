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


class Emb_AM_Dataset(Dataset):
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


class Features_MD_Dataset(Dataset):
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

if __name__ == '__main__':
    list_of_value_headers=["gram","gram2","gram3","gram4","mm","mm2","mm3","mm4","mm5","mm6","cm","cm2","cm3","cm4","cm5","cm6","cm7","cm8","cm9","cm10","cm11","cm12","cm13","cm14","cm15","cm16","cm17","cm18","cm19","cm20","cm21","cm22","cm23","cm24","cm25","cm26","cm27","cm28","cm29","cm30","ml","gr","gr2","gr3","gr4","gr5","gr6","gr7","gr8","kg","kg2","kg3","G","G2","gr.","m","liter"]
    ds = Features_MD_Dataset("data/AM_entities_units_w_emb.csv", list_of_value_headers=list_of_value_headers)
    # ds = Emb_AM_Dataset("data/entities_md_raw_VALIDATION_w_emb.csv")
    dataloader = DataLoader(ds, batch_size=1)

    print("n_classes:", ds.get_n_classes())
    # print("embedding_size:", ds.get_embedding_size())

    for sample in dataloader:
        X = sample['embedding']
        print(X)
        y = sample['target_class']
        # print("Shape of X [N, C, H, W]: ", X.shape)
        # print("Shape of y: ", y.shape, y.dtype)
        print(X.shape[1])
        break