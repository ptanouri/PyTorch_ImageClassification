import torch
# from PIL import Image
import torch.nn as nn
from torch import save, load

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import math


# training loop

class WineDataset:
    def __init__(self, transform = None):
        # data loading
        xy = np.loadtxt('../data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples. 1
        self.n_samples = xy.shape[0]
        
        self.transform = transform
        # data = datasets.load_wine()
        # self.x = torch.from_numpy(data.data.astype(np.float32))
        # self.y = torch.from_numpy(data.target.astype(np.float32))
        # self.y = self.y.view(self.y.shape[0], 1)
        # self.n_samples, self.n_features = self.x.shape

    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
    
    
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))


composed = transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]

features, labels = first_data
print(type(features), type(labels))
print(features)
# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# #training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples/4)
# print(total_samples, n_iterations)

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         # forward, backward, update
#         if (i+1) % 5 == 0:
#             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')


# # dataiter = iter(dataloader)
# # if dataiter:
# #     data = next(dataiter)
#     features, labels = data
#     print(features, labels)
# else:
#     print("DataLoader is empty.")

