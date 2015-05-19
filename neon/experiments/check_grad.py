# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
"""
Numerical gradient checking to validate backprop code.
"""

import logging
import numpy as np

from neon.datasets.synthetic import UniformRandom
from neon.experiments.experiment import Experiment
from neon.models.mlp import MLP
from neon.util.compat import range


logger = logging.getLogger(__name__)


class GradientChecker(Experiment):
    """
    In this `Experiment`, a model is trained on a fake training dataset to
    validate the backprop code within the given model.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def transfer(self, experiment):
        self.model = experiment.model
        self.dataset = experiment.dataset

    def save_state(self):
        for ind in range(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            self.weights[ind][:] = layer.weights

    def load_state(self):
        for ind in range(len(self.trainable_layers)):
            layer = self.model.layers[self.trainable_layers[ind]]
            layer.weights[:] = self.weights[ind]

    def check_layer(self, layer, inputs, targets):
        # Check up to this many weights.
        nmax = 30
        if type(layer.updates) == list:
            updates = layer.updates[0].asnumpyarray().ravel()
        else:
            updates = layer.updates.asnumpyarray().ravel()
        weights = layer.weights.asnumpyarray().ravel()
        grads = np.zeros(weights.shape)
        inds = np.random.choice(np.arange(weights.shape[0]),
                                min(weights.shape[0], nmax),
                                replace=False)
        for ind in inds:
            saved = weights[ind]
            weights[ind] += self.eps
            self.model.fprop(inputs)
            cost1 = self.model.cost.apply_function(targets).asnumpyarray()

            weights[ind] -= 2 * self.eps
            self.model.fprop(inputs)
            cost2 = self.model.cost.apply_function(targets).asnumpyarray()

            grads[ind] = ((cost1 - cost2) / self.model.layers[-1].batch_size *
                          layer.learning_rule.learning_rate / (2 * self.eps))
            weights[ind] = saved

        grads -= updates
        diff = np.linalg.norm(grads[inds]) / nmax
        if diff < 0.0002:
            logger.info('diff %g. layer %s OK.', diff, layer.name)
            return True

        logger.error('diff %g. gradient check failed on layer %s.',
                     diff, layer.name)
        return False

    def check_layerb(self, layer):
        # Check up to this many weights.
        nmax = 30
        if type(layer.updates) == list:
            updates = layer.updates[0].asnumpyarray().ravel()
        else:
            updates = layer.updates.asnumpyarray().ravel()
        weights = layer.weights.asnumpyarray().ravel()
        grads = np.zeros(weights.shape)
        inds = np.random.choice(np.arange(weights.shape[0]),
                                min(weights.shape[0], nmax),
                                replace=False)
        for ind in inds:
            saved = weights[ind]
            weights[ind] += self.eps
            self.model.data_layer.reset_counter()
            self.model.fprop()
            cost1 = self.model.cost_layer.get_cost().asnumpyarray()

            weights[ind] -= 2 * self.eps
            self.model.data_layer.reset_counter()
            self.model.fprop()
            cost2 = self.model.cost_layer.get_cost().asnumpyarray()

            grads[ind] = ((cost1 - cost2) / self.model.batch_size *
                          layer.learning_rule.learning_rate / (2 * self.eps))
            weights[ind] = saved

        grads -= updates
        diff = np.linalg.norm(grads[inds]) / nmax
        if diff < 0.0002:
            logger.info('diff %g. layer %s OK.', diff, layer.name)
            return True

        logger.error('diff %g. gradient check failed on layer %s.',
                     diff, layer.name)
        return False

    def run(self):
        """
        Actually carry out each of the experiment steps.
        """
        if not (hasattr(self.model, 'fprop') and hasattr(self.model, 'bprop')):
            logger.error('Config file not compatible.')
            return

        self.eps = 1e-4
        self.weights = []
        self.trainable_layers = []
        for ind in range(len(self.model.layers)):
            layer = self.model.layers[ind]
            if not (hasattr(layer, 'weights') and hasattr(layer, 'updates')):
                continue
            self.weights.append(layer.backend.copy(layer.weights))
            self.trainable_layers.append(ind)

        if not hasattr(layer, 'dataset'):
            if isinstance(self.model, MLP):
                datashape = (self.model.data_layer.nout,
                             self.model.cost_layer.nin)
            else:
                datashape = (self.model.layers[0].nin,
                             self.model.layers[-1].nout)
            self.dataset = UniformRandom(self.model.batch_size,
                                         self.model.batch_size,
                                         datashape[0], datashape[1])
            self.dataset.set_batch_size(self.model.batch_size)
            self.dataset.backend = self.model.backend
            self.dataset.load()
        ds = self.dataset

        if isinstance(self.model, MLP):
            self.model.data_layer.dataset = ds
            self.model.data_layer.use_set('train')
            self.model.fprop()
            self.model.bprop()
            self.model.update(0)

            self.save_state()
            self.model.data_layer.reset_counter()
            self.model.fprop()
            self.model.bprop()
            self.model.update(0)
            self.load_state()
        else:
            inputs = ds.get_batch(ds.get_inputs(train=True)['train'], 0)
            targets = ds.get_batch(ds.get_targets(train=True)['train'], 0)

            self.model.fprop(inputs)
            self.model.bprop(targets, inputs)
            self.model.update(0)

            self.save_state()
            self.model.fprop(inputs)
            self.model.bprop(targets, inputs)
            self.model.update(0)
            self.load_state()

        for ind in self.trainable_layers[::-1]:
            layer = self.model.layers[ind]
            if isinstance(self.model, MLP):
                result = self.check_layerb(layer)
            else:
                result = self.check_layer(layer, inputs, targets)
            if result is False:
                break

def ExhaustiveGradientChecker(layer, params, updates, in_dim, in_array=None):
    """
    Exhaustivly check that the numerical gradients match up with the bprop method
    for a layer. Should be used with toy layer sizes.

    Parameters
    ----------
    layer : layer object
        Should contain fprop and bprop methods along with output and delta attributes.
    params : tuple
        Tuple containing pointers to layer parameter arrays.
    updates : tuple
        Tuple containing points to layer parameter updates set by bprop.
    in_dim : tuple
        Expected shape for fprop.
    """
    deltaX = 1.e-6
    rng = np.random.RandomState(0) #change this to neon friendly
    fprop = layer.fprop
    bprop = layer.bprop

    if in_array is not None:
        in_center = in_array.copy()
    else:
        in_center = rng.randn(in_dim)
    fprop(in_center)
    out_center = layer.output.asnumpyarray()

    def num_grad(f_plus, f_minus, deltaX):
        return (f_plus-f_finus)/(2.*deltaX)

    def check_param(param, update, layer, in_array, out_array):
        orig_param = param.asnumpyarray()
        fprop = layer.fprop
        bprop = layer.bprop

        param_shape = param.shape
        update_shape = update.shape
        in_shape = in_array.shape
        out_shape = out_array.shape

        for ii in xrange(np.prod(param_shape)):
            for jj in xrange(np.prod(out_shape)):
                # Pick the jjth delta
                param[:] = orig_param.copy()
                layer.fprop(in_array)
                deltas_in = np.zeros(np.prod(out_shape))
                delta_in[jj] = 1.
                delta_in = delta_in.reshape(out_shape)
                layer.bprop(delta_in)
                exact = update.asnumpyarray.ravel()[ii]
                #Vary the iith param
                flat_param = orig_param.ravel()
                flat_param[ii] = param[ii]+deltaX
                param[:] = flat_param.reshape(param_shape)
                fprop(in_array)
                fplus = layer.output.ravel()[jj]
                flat_param[ii] = param[ii]-2.*deltaX
                param[:] = flat_param.reshape(param_shape)
                fprop(in_array)
                fminus = layer.output.ravel()[jj]
                if not np.allclose(exact, num_grad(fplus, fminus, deltaX)):
                    raise ValueError('Bad gradient in layer: '+str(layer.name)+'.')

    def check_input(layer, deltas, in_array, out_array):
        orig_in_array = in_array.asnumpyarray()
        fprop = layer.fprop
        bprop = layer.bprop

        param_shape = param.shape
        update_shape = update.shape
        in_shape = in_array.shape
        out_shape = out_array.shape

        for ii in xrange(np.prod(input_shape)):
            for jj in xrange(np.prod(out_shape)):
                # Pick the jjth delta
                layer.fprop(orig_in_array)
                deltas_in = np.zeros(np.prod(out_shape))
                delta_in[jj] = 1.
                delta_in = delta_in.reshape(out_shape)
                layer.bprop(delta_in)
                exact = deltas.asnumpyarray.ravel()[ii]
                #Vary the iith input
                flat_input = orig_input_array.ravel()
                flat_input[ii] = flat_input[ii]+deltaX
                in[:] = flat_input.reshape(in_shape)
                fprop(in_array)
                fplus = layer.output.ravel()[jj]
                flat_input[ii] = input[ii]-2.*deltaX
                input[:] = flat_input.reshape(in_shape)
                fprop(in_array)
                fminus = layer.output.ravel()[jj]
                if not np.allclose(exact, num_grad(fplus, fminus, deltaX)):
                    raise ValueError('Bad gradient in layer: '+str(layer.name)+'.')

                
    match = [check_param(param, update, layer, in_center, out_center)
             for param, update in zip(params, updates)]
    deltas_match = check_input(layer, deltas, in_array, out_array)
    return (all(match) and deltas_match)

