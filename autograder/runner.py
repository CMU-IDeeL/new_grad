import numpy as np

# NOTE: If you are on Windows and are having trouble with imports, try to run 
# this file from inside the autograder directory.
import sys
sys.path.append('./..')
sys.path.append('mytorch')

import mytorch
from test_linear import *
from test_autograd import *
from test_activation import *
from test_loss import *
from test_functional import *

tests = [
    {
        'name': '0.1 - Autograd Add Operation',
        'autolab': 'Autograd Add Operation',
        'handler': test_add_operation,
        'value': 5,
    },
    {
        'name': '0.2 - Autograd Backward',
        'autolab': 'Autograd Backward',
        'handler': test_backward,
        'value': 5,
    },
    {
        'name': '1.1 - Functional Backward - Multiply',
        'autolab': 'Functional Backward - Multiply',
        'handler': test_mul_backward,
        'value': 1,
    },
    {
        'name': '1.2 - Functional Backward - Subtraction',
        'autolab': 'Functional Backward - Subtraction',
        'handler': test_sub_backward,
        'value': 1,
    },
    {
        'name': '1.3 - Functional Backward - Matmul',
        'autolab': 'Functional Backward - Matmul',
        'handler': test_matmul_backward,
        'value': 1,
    },
    {
        'name': '1.4 - Functional Backward - Divide',
        'autolab': 'Functional Backward - Divide',
        'handler': test_div_backward,
        'value': 1,
    },
    {
        'name': '1.5 - Functional Backward - Log',
        'autolab': 'Functional Backward - Log',
        'handler': test_log_backward,
        'value': 1,
    },
    {
        'name': '1.6 - Functional Backward - Exp',
        'autolab': 'Functional Backward - Exp',
        'handler': test_exp_backward,
        'value': 1,
    },
    {
        'name': '2.1 - Linear (Autograd) Forward',
        'autolab': 'Linear (Autograd) Forward',
        'handler': test_linear_layer_forward,
        'value': 2,
    },
    {
        'name': '2.2 - Linear (Autograd) Backward',
        'autolab': 'Linear (Autograd) Backward',
        'handler': test_linear_layer_backward,
        'value': 4,
    },
    {
        'name': '2.3 - Linear + Skip Connection (Autograd) Forward',
        'autolab': 'Linear + Skip Connection (Autograd) Forward',
        'handler': test_linear_skip_forward,
        'value': 1,
    },
    {
        'name': '2.4 - Linear + Skip Connection (Autograd) Backward',
        'autolab': 'Linear + Skip Connection (Autograd) Backward',
        'handler': test_linear_skip_backward,
        'value': 2,
    },
    {
        'name': '3.1 - Identity (Autograd) Forward',
        'autolab': 'Identity (Autograd) Forward',
        'handler': test_identity_forward,
        'value': 1,
    },
    {
        'name': '3.2 - Identity (Autograd) Backward',
        'autolab': 'Identity (Autograd) Backward',
        'handler': test_identity_backward,
        'value': 4,
    },
    {
        'name': '3.3 - Sigmoid (Autograd) Forward',
        'autolab': 'Sigmoid (Autograd) Forward',
        'handler': test_sigmoid_forward,
        'value': 1,
    },
    {
        'name': '3.4 - Sigmoid (Autograd) Backward',
        'autolab': 'Sigmoid (Autograd) Backward',
        'handler': test_sigmoid_backward,
        'value': 4,
    },
    {
        'name': '3.5 - Tanh (Autograd) Forward',
        'autolab': 'Tanh (Autograd) Forward',
        'handler': test_tanh_forward,
        'value': 1,
    },
    {
        'name': '3.6 - Tanh (Autograd) Backward',
        'autolab': 'Tanh (Autograd) Backward',
        'handler': test_tanh_backward,
        'value': 4,
    },
    {
        'name': '3.7 - ReLU (Autograd) Forward',
        'autolab': 'ReLU (Autograd) Forward',
        'handler': test_relu_forward,
        'value': 1,
    },
    {
        'name': '3.8 - ReLU (Autograd) Backward',
        'autolab': 'ReLU (Autograd) Backward',
        'handler': test_relu_backward,
        'value': 4,
    },
    {
        'name': '4.1 - SoftmaxCrossEntropy Loss (Autograd) Forward',
        'autolab': 'SoftmaxCrossEntropy Loss (Autograd) Forward',
        'handler': test_softmaxXentropy_forward,
        'value': 1,
    },
    {
        'name': '4.2 - SoftmaxCrossEntropy Loss (Autograd) Backward',
        'autolab': 'SoftmaxCrossEntropy Loss (Autograd) Backward',
        'handler': test_softmaxXentropy_backward,
        'value': 4,
    }
]
tests.reverse()


if __name__=='__main__':
    run_tests(tests)


