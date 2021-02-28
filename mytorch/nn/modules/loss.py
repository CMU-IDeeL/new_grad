import numpy as np
from mytorch.nn.functional import matmul_backward, add_backward, sub_backward, mul_backward, div_backward

class MSELoss:
    def __init__(self, autograd):
        self.autograd = autograd
        self.loss_val = None

    def __call__(self, y, y_hat):
        self.forward(y, y_hat)

    # TODO: Use your working MSELoss forward and add operations to 
    # autograd_engine.
    def forward(self, y, y_hat):
        """
            This class is similar to the wrapper functions for the activations
            that you wrote in functional.py with a couple of key differences:
                1. Notice that instead of passing the autograd object to the forward
                    method, we are instead saving it as a class attribute whenever
                    an MSELoss() object is defined. This is so that we can directly 
                    call the backward() operation on the loss as follows:
                        >>> mse_loss = MSELoss(autograd_object)
                        >>> mse_loss(y, y_hat)
                        >>> mse_loss.backward()

                2. Notice that the class has an attribute called self.loss_val. 
                    You must save the calculated loss value in this variable and 
                    the forward() function is not expected to return any value.
                    This is so that we do not explicitly pass the divergence to 
                    the autograd engine's backward method. Rather, calling backward()
                    on the MSELoss object will take care of that for you.

            Args:
                - y (np.ndarray) : the ground truth,
                - y_hat (np.ndarray) : the output computed by the network,

            Returns:
                - No return required
        """
        #TODO: Use the primitive operations to calculate the MSE Loss
        #TODO: Remember to use add_operation to record these operations in
        #      the autograd engine after each operation

    def backward(self):
        # You can call autograd's backward here or in the mlp.
        raise NotImplementedError

# Hint: To simplify things you can just make a backward for this loss and not
# try to do it for every operation.
class SoftmaxCrossEntropy:
    def __init__(self):
        self.loss_val = None
        self.y_grad_placeholder = None

    def __call__(self, y, y_hat, autograd):
        return self.forward(y, y_hat, autograd)

    def forward(self, y, y_hat, autograd):
        """
            Refer to the comments in MSELoss
        """
        raise NotImplementedError


    def backward(self, autograd):
        # You can call autograd's backward here or in the mlp.
        raise NotImplementedError
