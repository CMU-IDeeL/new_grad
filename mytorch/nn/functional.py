import numpy as np
from mytorch.autograd_engine import Autograd

# Mathematical Functionalities needed
# also make sure grad of inputs are exact same shape as inputs, accounting for broadcasting.
def add_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = grad_output
    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    raise NotImplementedError


def mul_backward(grad_output, a, b):
    raise NotImplementedError


def sub_backward(grad_output, a, b):
    raise NotImplementedError


def div_backward(grad_output, a, b):
    raise NotImplementedError


def exp_backward(grad_output,a):
    raise NotImplementedError


def log_backward(grad_output, a):
    raise NotImplementedError
