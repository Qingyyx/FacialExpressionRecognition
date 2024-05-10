from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class RAF(data.Dataset):

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('./data/RAF_data.h5', 'r', driver='core')
        if self.split == 'Training':
            self.train_data = self.data['data_train_img']
            self.train_labels = self.data['data_train_label']

        elif self.split == 'Testing':
            self.test_data = self.data['data_test_img']
            self.test_labels = self.data['data_test_label']

    def __getitem__(self, index):
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)
