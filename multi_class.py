import torch
# from PIL import Image
import torch.nn as nn
from torch import save, load
import numpy as np
import matplotlib.pyplot as plt

#Multipleclass problem
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
  
  
#MULTI_CLASS      
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    
model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
# loss and optimizer
criteria = nn.CrossEntropyLoss()    


#########################

#BINARY_CLASS

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)
        return y_pred

model1 = NeuralNet1(input_size=28*28, hidden_size=5, num_classes=1)
criteria1 = nn.BCELoss()
    



