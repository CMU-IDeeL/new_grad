# new_grad

### Introduction
Most modern machine learning and deep learning frameworks rely on a technique called "[Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)" (or Autodiff for short) to compute gradients. In this homework assignment, we introduce a new Autodiff based framework for computing these gradients (called new_grad) with a focus on the backbone of the autodiff framework - the Autograd Engine - without the complexity of dealing with a special "Tensor" class, or the need to perform DFS during the backward pass over the computational graph.

### Learning Objectives
In this (optional) bonus homework assignment, you will biuld your own version of an automatic differentiation library, in the context of the Deep Learning concepts that you learn in the [course](http://deeplearning.cs.cmu.edu/S21/index.html).
What you will learn:
1. What is a computational graph? How are mathematical operations recorded on a computational graph?
2. The benefits of thinking at the granularity of operations, instead of layers,
3. The simplicity of the chain rule of differentiation, and backpropogation,
4. Building a PyTorch-like API without the use of a 'Tensor' class, and instead working directly with numpy arrays,

### Instructions
Though we provide some starter code, you will get to complete key components of the library including the main Autograd engine. For specific instructions, please refer to the writeup included in this repository. Students enrolled in the course: submit your solutions through Autolab.

### A quick demo of the API
Once you have completed all the key components of the assignment, you will be able to build and train simple neural networks:

1. Import mytorch and the nn module:
```Python3
>>> from mytorch import autograd_engine
>>> import mytorch.nn as nn
```
2. Declare the autograd object, layers, activations, and loss:
```Python3
>>> autograd_engine = autograd_engine.Autograd()
>>> linear1 = nn.Linear(input_shape, output_shape, autograd_engine)
>>> activation1 = nn.ReLU(autograd_engine)
>>> loss = nn.SoftmaxCrossEntropy(autograd_engine)
```

3. Calculate the loss, and kick-off backprop
```Python3
>>> y_hat = activation1(linear1(x))
>>> loss(y, y_hat)
>>> loss.backward()
```
 -------------------------
 Developed by: Kinori Rosnow, David Park, Anurag Katakkar, Shriti Priya, Shayeree Sarkar, and Bhiksha Raj.
 
