#1) Design model(input, output size, forward pass)
#2) Construct loss and optimizer
#3) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights



import torch
# from PIL import Image
import torch.nn as nn
# from torch import save, load
# from torch.optim import Adam
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 0) prepare data
X_numpy, y_numpy = make_regression(n_samples=100, n_features=1, noise=20, random_state=5)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
print(n_samples, n_features)

#1)model

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


#2)loss and optimizer

learning_rate = 0.01
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3) training loop

num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_pred = model(X)
    loss = criteria(y_pred, y)
    
    #backward pass
    loss.backward()
    
    #update
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')

plt.plot(X_numpy, predicted, 'b')

plt.show()