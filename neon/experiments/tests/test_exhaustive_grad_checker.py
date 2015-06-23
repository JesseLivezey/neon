from neon.backends import cpu
from neon.experiments.check_grad import exhaustive_gradient_checker
import numpy as np

def test_exhaustive_gradient_checker():
    class quadratic_layer(object):
        def __init__(self, a, inputs, backend):
            class pl(object):
                def __init__(self, output):
                    self.output = output

            self.a = a
            self.prev_layer = pl(inputs)
            self.backend = backend
            self.nin = inputs.shape[0]
            self.nout = inputs.shape[0]
            self.batch_size = inputs.shape[1]
            self.output = be.zeros(inputs.shape)
            self.params = [self.a]
            self.updates = [backend.zeros(self.a.shape)]
            self.deltas = backend.zeros(inputs.shape)
            self.temp_mat = backend.zeros(inputs.shape)
            self.name = 'quad'
        def fprop(self, inputs):
            be = self.backend
            be.power(inputs, 2, self.temp_mat)
            be.multiply(self.a, self.temp_mat, self.output)
        def bprop(self, deltas):
            inputs = self.prev_layer.output
            be = self.backend
            be.multiply(deltas, self.temp_mat, self.temp_mat)
            be.sum(self.temp_mat, (0,1), self.updates[0])
            be.multiply(self.a, inputs, self.deltas)
            be.multiply(self.deltas, deltas, self.deltas)
            be.multiply(2., self.deltas, self.deltas)

    be = cpu.CPU(default_dtype=np.float64)
    a = be.array(2.)
    inputs = be.array(np.arange(15).reshape(3,5))
    layer = quadratic_layer(a, inputs, be)
    layer.fprop(inputs)
    exhaustive_gradient_checker(layer)

