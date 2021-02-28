import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.nn.functional import *
    
# Activation Layer test - for Sigmoid, Tanh and ReLU

def test_identity():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x, autograd)
    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    test_act = nn.Identity()
    a1_out = test_act(l1_out, autograd)
    autograd.backward(1)
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    
    return True

def test_sigmoid():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x, autograd)
    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    test_act = nn.Sigmoid()
    a1_out = test_act(l1_out, autograd)
    autograd.backward(1)
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    
    return True

def test_tanh():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x, autograd)
    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    test_act = nn.Tanh()
    a1_out = test_act(l1_out, autograd)
    autograd.backward(1)
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    
    return True

def test_relu():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x, autograd)
    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.ReLU()
    test_act = nn.ReLU()
    a1_out = test_act(l1_out, autograd)
    autograd.backward(1)
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    compare_np_torch(a1_out, torch_a1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    
    return True