Neural Networks
Neural networks can be constructed using the torch.nn package.

Now that you had a glimpse of autograd, nn depends on autograd to define models and differentiate them. An nn.Module contains layers, and a method forward(input) that returns the output.

For example, look at this network that classifies digit images:

It is a simple feed-forward network. It takes the input, feeds it through several layers one after the other, and then finally gives the output.

A typical training procedure for a neural network is as follows:

Define the neural network that has some learnable parameters (or weights)

Iterate over a dataset of inputs

Process input through the network

Compute the loss (how far is the output from being correct)

Propagate gradients back into the network’s parameters

Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
