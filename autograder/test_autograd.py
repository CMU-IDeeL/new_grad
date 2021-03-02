import numpy as np
import torch
from mytorch.autograd_engine import Autograd, Operation
import mytorch.nn.functional as F
from helpers import compare_np_torch

def test_add_operation():
    autograd_engine = Autograd()
    x = np.random.randn(1, 5)
    y = np.random.randn(1, 5)

    z = x + y

    autograd_engine.add_operation(
        inputs=[x, y],
        output=z,
        gradients_to_update=[None, None],
        backward_operation=F.add_backward
    )

    assert len(autograd_engine.operation_list) == 1
    
    operation = autograd_engine.operation_list[0]
    assert type(operation) == Operation

    assert len(operation.inputs) == 2
    assert np.array_equal(operation.inputs[0], x) and np.array_equal(operation.inputs[1], y)
    assert np.array_equal(operation.output, z)
    assert len(operation.gradients_to_update) == 2
    assert operation.backward_operation == F.add_backward
    return True


def test_backward():
    autograd_engine = Autograd()
    x1 = np.random.randn(1, 5)
    y1 = np.random.randn(1, 5)

    z1 = x1 + y1

    autograd_engine.add_operation(
        inputs=[x1, y1],
        output=z1,
        gradients_to_update=[None, None],
        backward_operation=F.add_backward
    )

    assert len(autograd_engine.operation_list) == 1
    assert len(autograd_engine.memory_buffer.memory)==2

    autograd_engine.backward(1)
    dy1 = autograd_engine.memory_buffer.get_param(y1)

    torch_x1 = torch.DoubleTensor(torch.tensor(x1, requires_grad=True))
    torch_y1 = torch.DoubleTensor(torch.tensor(y1, requires_grad=True))
    torch_x1.retain_grad()
    torch_y1.retain_grad()

    torch_z1 = torch_x1 + torch_y1
    torch_z1.sum().backward()

    compare_np_torch(z1, torch_z1)
    compare_np_torch(dy1, torch_y1.grad)
    compare_np_torch(autograd_engine.memory_buffer.get_param(x1), torch_x1.grad)
    return True
