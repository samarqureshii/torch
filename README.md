# torch
```
pip3 install torch
pip3 install numpy
```

> Intro to tensors, autogradients, neural networks, and training a simple classifier.
- `torch.Tensor` - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
- `nn.Module` - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
- `nn.Parameter` - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
- `autograd.Function` - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.

