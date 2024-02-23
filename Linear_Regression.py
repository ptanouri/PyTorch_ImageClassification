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
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt


# x = torch.rand(3, requires_grad=True)




# print(x)
# # print(x)

# y = x+2

# print(y)

# z = y*y*2

# # z = z.mean() #dz/dx
# print(z)

# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

# z.backward(v)

# print(x.grad)   

# weights = torch.ones(4, requires_grad=True)


# # for epoch in range(3):
# #     model_output = (weights*3).sum()
# #     model_output.backward()
# #     print(weights.grad)
# #     weights.grad.zero_() #reset the gradients to zero
    

# optimzer = torch.optim.SGD([weights], lr=0.01)
# optimzer.step()
# optimzer.zero_grad()

# x = torch.tensor(1.0)

# y=torch.tensor(2.0)

# w = torch.tensor(1.0, requires_grad=True)

# #forward pass and compute the loss
# y_hat = w*x
# loss = (y_hat-y)**2

# print(loss)

# #backward pass
# loss.backward()
# print(w.grad)
# print('hi')

#f = 2*x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)

Y = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features

output_size = n_features

# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

#loss = MSE



#gradient

#MSE = 1/N * (w*x - y)**2
#dJ/dw = 1/N 2x(w*x - y)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

#Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(X)

    #loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    l.backward() #dl/dw

    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    


    if epoch % 10 == 0:
        
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0].item():.3f}, loss = {l:.8f}')
        
        
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')