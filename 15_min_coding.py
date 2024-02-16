import torch
import torch.nn as nn
from torch.optim import adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt



#Get Data
train = datasets.MNIST(root = "data", download=True, train = True, transform = ToTensor())
dataset = DataLoader(train, 32)


# Move model to CPU
model = model.cpu()

# Move tensor to CPU
tensor = tensor.cpu()

#Image Classifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()    
        self.model = nn.Sequential(nn.Conv2d(1,32,(3,3)), nn.ReLU(),nn.Conv2d(32,64,(3,3)), nn.ReLU(),nn.Conv2d(64,64,(3,3)), nn.ReLU(), nn.Flatten(), 
                                   nn.Linear(28*28, 128), nn.Linear(64*(28-6)*(28-6), 10))

        # super(ImageClassifier, self).__init__()
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(28*28, 128)
        # self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        
        # x = self.flatten(x)
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        return self.model(x)
    
    #instance of the neural network, optimizer and loss function
    
clf = ImageClassifier().to('cuda')
optimizer = adam(clf.parameters(), lr = 0.001)
    
loss_func = nn.CrossEntropyLoss()
    
    #Train the model
    
if __name__ == "__main__":
    for epoch in range(10):
        for batch, (X, y) in enumerate(dataset):
            
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            
            y_pred = clf(X)
            loss = loss_func(y_pred, y)
            
            #apply backpropagation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
        
    with open('model.pt', 'wb') as f:
        torch.save(clf, f)
