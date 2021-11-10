import numpy as np

import sys
sys.path.append('..')
import os

from mytorch import autograd_engine
import mytorch.nn as nn
from mytorch import optim
from mytorch.nn.functional import *

DATA_PATH = "./data"

class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations,
                 criterion, lr, autograd_engine, momentum=0.0):

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.autograd_engine = autograd_engine # NOTE: Use this Autograd object for backward
        self.linear_layers = None

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        raise NotImplementedError

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear layers.
        raise NotImplementedError

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        raise NotImplementedError

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        raise NotImplementedError
        # NOTE: Put the inputs in the correct order for the criterion
        # return self.criterion().sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size=1):
    # NOTE: Because the batch size is 1 (unless you support
    # broadcasting) the MLP training will be slow.
    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    for e in range(nepochs):

        # Per epoch setup ...
        for b in range(0, len(trainx)):
            # Train ...
            # NOTE: Batchsize is 1 for this bonus unless you support 
            # broadcasting/unbroadcasting then you can change this in
            # the mlp_runner.py
            x = np.expand_dims(trainx[b], 0)
            y = np.expand_dims(trainy[b], 0)

        for b in range(0, len(valx)):
            # Val ...
            x = np.expand_dims(valx[b], 0)
            y = np.expand_dims(valy[b], 0)

        # Accumulate data...

    # Cleanup ...

    # Return results ...
    return (training_losses, training_errors, validation_losses, validation_errors)


def load_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y
