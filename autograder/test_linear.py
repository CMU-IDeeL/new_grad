import numpy as np
import torch

from mytorch import autograd_engine
import mytorch.nn as nn
from helpers import *
from mytorch.nn.functional import *

def test_linear_layer():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))  # just use batch size 1 since broadcasting is not written yet

    l1_out = l1(x)
    autograd.backward(1)
    
    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))  # note transpose here, probably should standardize
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))
    torch_x = torch.DoubleTensor(x)
    torch_l1_out = torch_l1(torch_x)
    torch_l1_out.sum().backward()
    compare_np_torch(l1_out, torch_l1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)  # transpose here too, because torch's weights are out x in
    compare_np_torch(l1.db, torch_l1.bias.grad)
    return True

def test_linear_skip():
    np.random.seed(0)
    autograd = autograd_engine.Autograd()

    autograd.zero_grad()
    l1 = nn.Linear(5, 5, autograd)
    x = np.random.random((1, 5))
    l1_out = l1(x)
    output = l1_out + x
    autograd.add_operation(inputs=[l1_out, x], output=output, gradients_to_update=[None, None],
                      backward_operation=add_backward)
    autograd.backward(1)

    torch_l1 = torch.nn.Linear(5, 5)
    torch_l1.weight = torch.nn.Parameter(torch.DoubleTensor(l1.W.T))
    torch_l1.bias = torch.nn.Parameter(torch.DoubleTensor(l1.b))

    torch_x = torch.DoubleTensor(x)
    torch_x.requires_grad = True

    torch_l1_out = torch_l1(torch_x)
    torch_output = torch_l1_out + torch_x
    torch_output.sum().backward()

    compare_np_torch(l1_out, torch_l1_out)
    compare_np_torch(l1.dW, torch_l1.weight.grad.T)
    compare_np_torch(l1.db, torch_l1.bias.grad)
    compare_np_torch(autograd.memory_buffer.get_param(x), torch_x.grad)  # skip connections work'''
    return True
