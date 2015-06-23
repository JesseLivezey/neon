from neon import layers, transforms
from neon import transforms
from neon.experiments.check_grad import exhaustive_gradient_checker
from neon.backends import cpu
from neon.util.defaults import default_weight_init
from neon.params.val_init import GaussianValGen
import numpy as np

class dummy_layer(object):
    def __init__(self, array):
        self.output = array
        self.is_local = False
        self.is_data = False
        self.nout = array.shape[0]

def test_FC_grad():
    batch_size = 2
    nin = 3
    nout = 5
    rng = np.random.RandomState(20150618)
    inpt = rng.randn(nin, batch_size)

    be = cpu.CPU(default_dtype=np.float64)
    pl = dummy_layer(be.array(inpt))
    kwargs = {'backend': be,
              'batch_size': batch_size}

    fc_layer = layers.FCLayer(nin=nin,
                              nout=nout,
                              pre_act_dtype=np.float64,
                              output_dtype=np.float64,
                              deltas_dtype=np.float64,
                              weight_dtype=np.float64,
                              updates_dtype=np.float64,
                              backend_type=np.float64)
    fc_layer.set_weight_shape()
    fc_layer.set_previous_layer(pl)
    fc_layer.initialize(kwargs)
    fc_layer.set_deltas_buf(None, None)
    exhaustive_gradient_checker(fc_layer, deltaX=1.e-6)

def test_RectLin_grad():
    batch_size = 2
    nin = 3
    nout = 5
    rng = np.random.RandomState(20150618)
    inpt = rng.randn(nin, batch_size)

    be = cpu.CPU(default_dtype=np.float64)
    pl = dummy_layer(be.array(inpt))
    kwargs = {'backend': be,
              'batch_size': batch_size}

    fc_layer = layers.FCLayer(nin=nin,
                              nout=nout,
                              pre_act_dtype=np.float64,
                              output_dtype=np.float64,
                              deltas_dtype=np.float64,
                              weight_dtype=np.float64,
                              updates_dtype=np.float64,
                              backend_type=np.float64,
                              activation=transforms.RectLin())
    fc_layer.set_weight_shape()
    fc_layer.set_previous_layer(pl)
    fc_layer.initialize(kwargs)
    fc_layer.set_deltas_buf(None, None)
    exhaustive_gradient_checker(fc_layer)

def test_Logistic_grad():
    batch_size = 2
    nin = 3
    nout = 5
    rng = np.random.RandomState(20150618)
    inpt = rng.randn(nin, batch_size)

    be = cpu.CPU(default_dtype=np.float64)
    pl = dummy_layer(be.array(inpt))
    kwargs = {'backend': be,
              'batch_size': batch_size}

    fc_layer = layers.FCLayer(nin=nin,
                              nout=nout,
                              pre_act_dtype=np.float64,
                              output_dtype=np.float64,
                              deltas_dtype=np.float64,
                              weight_dtype=np.float64,
                              updates_dtype=np.float64,
                              backend_type=np.float64,
                              activation=transforms.Logistic())
    fc_layer.set_weight_shape()
    fc_layer.set_previous_layer(pl)
    fc_layer.initialize(kwargs)
    fc_layer.set_deltas_buf(None, None)
    exhaustive_gradient_checker(fc_layer)

def test_recurrent_hidden_grad():
    batch_size = 2
    nin = 3
    nout = 5
    time_steps = 7
    rng = np.random.RandomState(20150618)
    inpt = rng.randn(time_steps, nin, batch_size)

    be = gen_backend()
    pl = dummy_layer(be.array(inpt))
    kwargs = {'backend': be,
              'batch_size': batch_size,
              'weight_init_rec': GaussianValGen(loc=0.0, scale=0.01, backend=be),
              'unrolls': time_steps}

    rec_layer = layers.RecurrentHiddenLayer(nin=nin, nout=nout)
    rec_layer.set_weight_shape()
    rec_layer.set_previous_layer(pl)
    rec_layer.initialize(kwargs)
    rec_layer.set_deltas_buf(None, None)
    exhaustive_gradient_checker(rec_layer)
