"""
NOTE: These test cases do not check the correctness of your solution,
      only whether anything has been implemented in functional.py.
      You are free to add your own test cases for checking correctness
"""

import numypy as np

from mytorch.nn.functional import *

def test_add_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros_like(grad_output)
    b = np.zeros_like(grad_output)
    if add_backward(grad_output, a, b):
        return True

def test_sub_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros_like(grad_output)
    b = np.zeros_like(grad_output)
    if sub_backward(grad_output, a, b):
        return True

def test_matmul_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros((5, 5))
    b = np.zeros((1, 5))
    if matmul_backward(grad_output, a, b):
        return True

def test_mul_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros_like(grad_output)
    b = np.zeros_like(grad_output)
    if mul_backward(grad_output, a, b):
        return True

def test_div_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros((5, 5))
    b = np.zeros((1, 5))
    if div_backward(grad_output, a, b):
        return True

def test_log_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros((5, 5))
    if log_backward(grad_output, a):
        return True

def test_exp_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros((5, 5))
    if exp_backward(grad_output, a):
        return True

def test_pow_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros((5, 5))
    if pow_backward(grad_output, a):
        return True
