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
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples. 1
        self.n_samples = xy.shape[0]
        # data = datasets.load_wine()
        # self.x = torch.from_numpy(data.data.astype(np.float32))
        # self.y = torch.from_numpy(data.target.astype(np.float32))
        # self.y = self.y.view(self.y.shape[0], 1)
        # self.n_samples, self.n_features = self.x.shape

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
    
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')


# dataiter = iter(dataloader)
# if dataiter:
#     data = next(dataiter)
#     features, labels = data
#     print(features, labels)
# else:
#     print("DataLoader is empty.")

