import numpy as np
from mytorch.autograd_engine import Autograd

'''
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
''' 
def add_backward(grad_output, a, b):
    a_grad = grad_output
    b_grad = grad_output
    return a_grad, b_grad

def sub_backward(grad_output, a, b):
    raise NotImplementedError

def matmul_backward(grad_output, a, b):
    raise NotImplementedError

def mul_backward(grad_output, a, b):
    raise NotImplementedError

def div_backward(grad_output, a, b):
    raise NotImplementedError

def log_backward(grad_output, a):
    raise NotImplementedError

def exp_backward(grad_output,a):
    raise NotImplementedError

def pow_backward(grad_output, a):
    raise NotImplementedError

