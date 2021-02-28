import numpy as np

from mytorch.autograd_engine import Autograd, Operation
import mytorch.nn.functional as F


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

    y2 = np.random.randn(1, 5)
    dy2 = np.zeros_like(y2)
    z2 = z1 * y2

    autograd_engine.add_operation(
        inputs=[z1, y2],
        output=z2,
        gradients_to_update=[None, dy2],
        backward_operation=F.mul_backward
    )

    autograd_engine.backward(1)
    # TODO: finish checks here
    return True