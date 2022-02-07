import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.nn.functional import *
    
# Activation Layer test - for Sigmoid, Tanh and ReLU

def test_identity_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    
    return True


def test_identity_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Identity(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Identity()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True


def test_sigmoid_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)

    return True


def test_sigmoid_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Sigmoid(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Sigmoid()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True


def test_tanh_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    return True


def test_tanh_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.Tanh(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.Tanh()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()

    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True


def test_relu_forward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)

    compare_np_torch(a1_out, torch_a1_out)
    
    return True


def test_relu_backward():
    # Test input
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    test_act = nn.ReLU(autograd)
    a1_out = test_act(l1_out)
    autograd.backward(1)

    # Torch input
    torch_l1 = torch.nn.Linear(5,5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b.squeeze()))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_act = torch.nn.ReLU()
    torch_a1_out = torch_act(torch_l1_out)
    torch_a1_out.sum().backward()
    
    compare_np_torch(l1.dW, torch_l1.weight.grad)
    compare_np_torch(l1.db.squeeze(), torch_l1.bias.grad)
    
    return True
