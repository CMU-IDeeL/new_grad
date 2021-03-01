import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.nn.functional import *
    
# Activation Layer test - for Sigmoid, Tanh and ReLU

def test_softmaxXentropy():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    y = np.array([[0., 0., 1., 0., 0.]]) # 1. must be in same location as index for torch_y
    l1_out = l1(x)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_y = torch.LongTensor(np.array([2])) # this value must be same as 1 index in y
    torch_l1_out = torch_l1(torch_x)

    test_loss = nn.SoftmaxCrossEntropy(autograd)
    a1_out = test_loss(y, l1_out)
    test_loss.backward()

    torch_loss = torch.nn.CrossEntropyLoss()
    print(torch_y, torch_l1_out)
    torch_a1_out = torch_loss(torch_l1_out, torch_y).reshape(1,) # input, target
    torch_a1_out.backward()

    print(a1_out, torch_a1_out)
    print(a1_out.shape, torch_a1_out.shape)

    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    
    return True
