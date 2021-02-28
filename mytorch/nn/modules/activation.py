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

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self, autograd_engine):
        super(Identity, self).__init__()
        self.autograd_engine = autograd_engine

    def forward(self, x):
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__()
        self.autograd_engine = autograd_engine

    def forward(self, x):
        raise NotImplementedError

class Tanh(Activation):
    def __init__(self, autograd_engine):
        super(Tanh, self).__init__()
        self.autograd_engine = autograd_engine

    def forward(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def __init__(self, autograd_engine):
        super(ReLU, self).__init__()
        self.autograd_engine = autograd_engine

    def forward(self, x):
        raise NotImplementedError
