from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import pdb

import numpy as np
import tensorflow as tf
import copy

from graph.common import Graph, propdict

import vvn.models.losses as losses
import vvn.ops.utils as utils
from .base import Model

class Decoder(Model):

    def __init__(
            self,
            name,
            model_func=None,
            input_signature=['inputs'],
            time_shared=True,
            **model_params
    ):
        self.name = name
        self.input_signature = input_signature
        self.time_shared = time_shared
        super(Decoder, self).__init__(
            name=self.name, model_func=model_func, **model_params)

    def build_inputs(self, inputs, input_mapping):
        assert isinstance(inputs, (dict, OrderedDict))
        assert isinstance(input_mapping, dict)
        assert all((v in input_mapping.keys() for v in self.input_signature)), "Must pass one input per signature item in %s" % self.input_signature

        input_list = [inputs[input_mapping[nm]] for nm in self.input_signature]
        return input_list

    def rename_outputs(self, outputs):
        outputs = {
            self.name + '/' + k: outputs[k] for k in outputs.keys()}
        return outputs

    def __call__(self, inputs, train=True,
                 input_mapping={'inputs': 'features/outputs'},
                 rename=True, **kwargs):

        kwargs['train'] = train
        decoder_inputs = self.build_inputs(inputs, input_mapping) # list

        if self.time_shared: # merge batch and time dimensions temporarily
            B,T = decoder_inputs[0].shape.as_list()[0:2]
            decoder_inputs = [tf.reshape(inp, [B*T] + inp.shape.as_list()[2:])
                              for inp in decoder_inputs]

        outputs = super(Decoder, self).__call__(
            *decoder_inputs, **kwargs)
        if not isinstance(outputs, dict):
            assert isinstance(outputs, tf.Tensor)
            outputs = {'outputs': outputs}

        if self.time_shared: # expand batch and time dimensions
            outputs = {k: tf.reshape(out, [B,T] + out.shape.as_list()[1:]) for k,out in outputs.items()}
        outputs = self.rename_outputs(outputs)
        self.outputs = outputs
        return outputs
