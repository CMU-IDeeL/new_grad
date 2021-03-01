import numpy as np

# NOTE: If you are on Windows and are having trouble with imports, try to run 
# this file from inside the autograder directory.
import sys
sys.path.append('./..')

import mytorch
from test_linear import *
from test_autograd import *
from test_activation import *
from test_loss import *
from test_functional import *

tests = [
    # funcitonals will be 0.5
    {
        'name': '0.1 - Autograd Add Operation',
        'autolab': 'Autograd Add Operation',
        'handler': test_add_operation,
        'value': 2,
    },
    {
        'name': '0.2 - Autograd Backward',
        'autolab': 'Autograd Backward',
        'handler': test_backward,
        'value': 2,
    },
    {
        'name': '1.1 - Functional Backward - Multiply',
        'autolab': 'Functional Backward - Multiply',
        'handler': test_mul_backward,
        'value': 0.5,
    },
    {
        'name': '1.2 - Functional Backward - Subtraction',
        'autolab': 'Functional Backward - Subtraction',
        'handler': test_sub_backward,
        'value': 0.5,
    },
    {
        'name': '1.3 - Functional Backward - Matmul',
        'autolab': 'Functional Backward - Matmul',
        'handler': test_matmul_backward,
        'value': 0.5,
    },
    {
        'name': '1.4 - Functional Backward - Divide',
        'autolab': 'Functional Backward - Divide',
        'handler': test_div_backward,
        'value': 0.5,
    },
    {
        'name': '1.5 - Functional Backward - Log',
        'autolab': 'Functional Backward - Log',
        'handler': test_log_backward,
        'value': 0.5,
    },
    {
        'name': '1.6 - Functional Backward - Exp',
        'autolab': 'Functional Backward - Exp',
        'handler': test_exp_backward,
        'value': 0.5,
    },
    {
        'name': '1.7 - Functional Backward - Pow',
        'autolab': 'Functional Backward - Pow',
        'handler': test_pow_backward,
        'value': 0.5,
    },
    {
        'name': '2.1 - Linear (Autograd)',
        'autolab': 'Linear (Autograd)',
        'handler': test_linear_layer,
        'value': 3,
    },
    {
        'name': '2.2 - Linear + Skip Connection (Autograd)',
        'autolab': 'Linear + Skip Connection (Autograd)',
        'handler': test_linear_skip,
        'value': 3,
    },
    {
        'name': '2.3 - Identity (Autograd)',
        'autolab': 'Identity (Autograd)',
        'handler': test_identity,
        'value': 2,
    },
    {
        'name': '2.4 - Sigmoid (Autograd)',
        'autolab': 'Sigmoid (Autograd)',
        'handler': test_sigmoid,
        'value': 2,
    },
    {
        'name': '2.5 - Tanh (Autograd)',
        'autolab': 'Tanh (Autograd)',
        'handler': test_tanh,
        'value': 2,
    },
    {
        'name': '2.6 - ReLU (Autograd)',
        'autolab': 'ReLU (Autograd)',
        'handler': test_relu,
        'value': 2,
    },
    {
        'name': '2.7 - SoftmaxCrossEntropy Loss (Autograd)',
        'autolab': 'SoftmaxCrossEntropy Loss (Autograd)',
        'handler': test_softmaxXentropy,
        'value': 2,
    }
]
tests.reverse()


if __name__=='__main__':
    run_tests(tests)


