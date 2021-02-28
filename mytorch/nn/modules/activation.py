import numpy as np
from mytorch.nn.functional import *

class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self):
        self.state = None

    def __call__(self, x, autograd):
        return self.forward(x, autograd)

    def forward(self, x, autograd):
        raise NotImplementedError

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, autograd):
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, autograd):
        raise NotImplementedError

class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x, autograd):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x, autograd):
        raise NotImplementedError
