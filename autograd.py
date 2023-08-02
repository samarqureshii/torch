# torch.autograd is PyTorch’s automatic differentiation engine that powers neural network training

#Forward Propagation: 
# In forward prop, the NN makes its best guess about the correct output. 
# It runs the input data through each of its functions to make this guess.

#Backward Propagation: 
# In backprop, the NN adjusts its parameters proportionate to the error in its guess. 
# It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters 
# of the functions (gradients), and optimizing the parameters using gradient descent

import torch
from torchvision.models import resnet18, ResNet18_Weights, optim, nn
model = resnet18(weights=ResNet18_Weights.DEFAULT) #load pretrained resnet18 from pytorch
data = torch.rand(1, 3, 64, 64) # random data tensor to represent a single image with 3 channels w height/width of 64
labels = torch.rand(1, 1000) # corresponding label initialized to some random values

#run the input data through the model through each of its layers to make a prediction
prediction = model(data) # forward pass

# We use the model’s prediction and the corresponding label to calculate the error (loss). 
# The next step is to backpropagate this error through the network.
# Backward propagation is kicked off when we call .backward() on the error tensor. 
# Autograd then calculates and stores the gradients for each model parameter in the parameter’s .grad attribute.
loss = (prediction - labels).sum()
loss.backward() # backward pass

#Next, we load an optimizer, in this case SGD with a learning rate of 0.01 and momentum of 0.9. 
# We register all the parameters of the model in the optimizer.
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in .grad
optim.step() #gradient descent



### DIFFRENTIATION IN AUTOGRAD

#collects gradients
# We create two tensors a and b with requires_grad=True. 
# This signals to autograd that every operation on them should be tracked
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

#create another tensor from a and b, Q = 3a^3 - b^2
Q = 3*a**3 - b**2

#aggregate Q into a scalar and call backward implicitly
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

## COMPUTATIONAL GRAPH
# Conceptually, autograd keeps a record of data (tensors) & all executed operations (along with the resulting new tensors) 
# in a directed acyclic graph (DAG) consisting of Function objects. 
# In this DAG, leaves are the input tensors, roots are the output tensors. 
# By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule

### EXLUCSION FROM DAG

#torch.autograd tracks operations on all tensors which have their requires_grad flag set to True. F
# or tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.
#The output tensor of an operation will require gradients even if only a single input tensor has requires_grad=True

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")


#In a NN, parameters that don’t compute gradients are usually called frozen parameters. 
# It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters 
# (this offers some performance benefits by reducing autograd computations)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
    
# finetune the model on a new dataset with 10 labels. 
# In resnet, the classifier is the last linear layer model.fc. 
# We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier
model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)