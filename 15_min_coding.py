import torch
from PIL import Image
import torch.nn as nn
from torch import save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt



#Get Data
train = datasets.MNIST(root = "data", download=True, train = True, transform = ToTensor())
dataset = DataLoader(train, 32)



#Image Classifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()    
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)), nn.ReLU(),nn.Conv2d(32,64,(3,3)), nn.ReLU(),nn.Conv2d(64,64,(3,3)), nn.ReLU(), nn.Flatten()
            , nn.Linear(64*(28-6)*(28-6), 10)
                                   )

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
    
clf = ImageClassifier().to('cpu')
optimizer = Adam(clf.parameters(), lr = 0.001)
    
loss_func = nn.CrossEntropyLoss()
    
    #Train the model
    
if __name__ == "__main__":
    # with open('model.pt', 'rb') as f:
    #     clf.load_state_dict(load(f))
        
    # img = Image.open('6.png')
    # img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')
    
    
    # print(torch.argmax(clf(img_tensor)))
    
    
    
    for epoch in range(1):
        for batch in dataset:
            X,y = batch
            X, y = X.to('cpu'), y.to('cpu')
            
            y_pred = clf(X)
            loss = loss_func(y_pred, y)
            
            #apply backpropagation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
        
    with open('model.pt', 'wb') as f:
        torch.save(clf, f)

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())
