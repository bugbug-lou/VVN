from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import copy

from vvn.ops.vectorizing import DimensionDict, aggregate_mean_and_var_features, compute_attr_spatial_moments, compute_border_attributes, add_history_attributes
from vvn.ops.convolutional import mlp, conv, depth_conv, convnet_stem, build_convrnn_inputs, coordinate_ims

def vectorize_inputs_model(
        inputs,
        train_targets,
        train=True,
        inp_sequence_len=3,
        inp_size=[64,64],
        segments_key='segments',
        max_num_objects=8,
        aggregation_kwargs={},
        add_spatial_attributes=True,
        spatial_attribute_kwargs={},
        add_border_attributes=True,
        border_attribute_kwargs={},
        add_previous_attributes=[],
        history_times=None,
        mlp_kwargs=None,
        **model_kwargs):
    '''
    Takes inputs from a data provider that include ground truth segmentation and tracking;
    Vectorizes these feature inputs to construct a single-level graph where each node is one of the objects/bg

    inputs: dict of (attr, tensor) pairs taken from a data provider
            feature inputs of shape [B,T,H,W,C] get preprocessed into features to aggregate
            of shape [B,T,H',W',C']
            segmentation inputs get processed to become grouping masks [B,T,H',W'] <tf.int32>
            other inputs (such as camera params) may be used in the aggregation phase

    '''
    params = {} # required return by tfutils
    outputs = {}

    # convert the dict of input retinal arrays into a single tensor (may include depth, normals, flows, etc.)
    inp_feats, Dims = build_convrnn_inputs(
        inputs, inp_sequence_len, ntimes=1, time_dilation=1,
        num_temporal_splits=inp_sequence_len, static_mode=False, is_training=train,
        **model_kwargs)
    inp_feats = inp_feats[0] # normally returns a list of length time_dilation, so take first element of shape [BT,H,W,C]
    InpDims = Dims.copy(suffix='')

    # get dimensions and resize if necessary
    BT,H,W,C = inp_feats.shape.as_list()
    Hf,Wf = inp_size
    T = inp_sequence_len
    B = BT // T
    inp_feats = tf.image.resize_images(inp_feats, inp_size, method=1) # nearest neighbor

    # get the ground truth segmentation
    N = max_num_objects
    segments = inputs[segments_key]
    if segments.shape.as_list()[2:] != inp_size:
        segments = tf.reshape(inputs[segments_key], [B*T,H,W,-1])
        segments = tf.image.resize_images(segments, inp_size, method=1) # nearest neighbor
        segments = tf.reshape(segments, [B,T,Hf,Wf])
    valid_segments = segments < tf.cast(max_num_objects, tf.int32)
    segments = tf.where(valid_segments, segments, (max_num_objects-1)*tf.ones_like(segments)) # [B,T,Hf,Wf]

    # start "vectorizing": aggregate features over the segments
    inp_attrs, num_segments, Dims = aggregate_mean_and_var_features(
        inp_feats, segments, dimension_dict=Dims, max_segments=N, **aggregation_kwargs)
    hw_attrs = inp_attrs[...,-4:-2]
    area_attrs = inp_attrs[...,-2:-1]
    valid_attrs = inp_attrs[...,-1:]
    inview_attrs = tf.cast(area_attrs > (0.5 / (Hf*Wf)), tf.float32)
    # set the background to not be in view
    inview_attrs = tf.concat([tf.zeros_like(inview_attrs[:,0:1]), inview_attrs[:,1:]], axis=1)
    null_attrs = tf.zeros_like(valid_attrs[...,1:])

    # treat the input features as level_0 nodes
    ones = tf.ones_like(inp_feats[...,0:1])
    nodes_level_0 = InpDims.extend_vector([
        ('hw_centroids', coordinate_ims(BT,1,[Hf,Wf])[:,0]),
        ('areas', ones / (Hf*Wf)),
        ('valid', ones)], base_tensor=inp_feats)
    nodes_level_0 = tf.reshape(nodes_level_0, [BT,Hf*Wf,-1])

    # optionally append spatial moment attributes, treating input features as level_0_nodes
    spatial_attrs = null_attrs
    if add_spatial_attributes:
        spatial_attrs,_,_,_,SpaDims = compute_attr_spatial_moments(
            nodes_level_0, segments, tf.reshape(segments, [-1]), num_segments,
            nodes_dimension_dict=InpDims, max_parent_nodes=N, hw_attr='hw_centroids', hw_dims=[-4,-2],
            **spatial_attribute_kwargs)
        Dims.insert_from(SpaDims)
        inp_attrs = tf.concat([inp_attrs, spatial_attrs], axis=-1)
    print("spatial_attrs", spatial_attrs.shape.as_list())

    # optionally append border attributes
    nodes_2d = tf.concat([hw_attrs, area_attrs, inview_attrs], axis=-1)
    border_attrs = compute_border_attributes(
        nodes_2d, segments, hw_dims=[-4,-2],
        **border_attribute_kwargs
    ) if add_border_attributes else null_attrs
    print("border attrs", border_attrs.shape.as_list())
    Dims['borders'] = border_attrs.shape.as_list()[-1]
    inp_attrs = tf.concat([inp_attrs, border_attrs], axis=-1)

    if len(add_previous_attributes):
        inp_attrs = tf.reshape(inp_attrs, [B,T,N,-1])
        inp_attrs, Dims = add_history_attributes(
            inp_attrs, Dims, add_previous_attributes, prev_times=history_times)
        inp_attrs = tf.reshape(inp_attrs, [BT,N,-1])

    # optionally learn latent attributes
    print("inp attrs to mlp", inp_attrs.shape.as_list())
    if mlp_kwargs is not None:
        learned_attrs = mlp(inp_attrs, scope='attribute_mlp', **mlp_kwargs)
    else:
        learned_attrs = null_attrs

    # concat all the attrs into nodes in the customary format
    nodes = Dims.extend_vector([
        ('learned', learned_attrs),
        ('inview', inview_attrs),
        ('hw_centroids', hw_attrs),
        ('areas', area_attrs),
        ('valid', valid_attrs)
    ], base_tensor=inp_attrs)

    # revert nodes to [B,T,N,D] format
    outputs['nodes_level_0'] = nodes_level_0[:,tf.newaxis]
    outputs['nodes_level_1'] = nodes[:,tf.newaxis]
    outputs['segments_level_1'] = tf.reshape(segments, [BT,1,Hf,Wf])
    combine_temporal_splits(outputs, inp_sequence_len, 1)

    # track the dimensions
    outputs['dimensions_level_0'] = InpDims.copy(suffix='')
    outputs['dimensions_level_1'] = Dims.copy(suffix='')

    print("nodes", nodes.shape.as_list())
    print("Dims final", Dims.sort())

    return outputs, params

def combine_temporal_splits(output_dict, num_temporal_splits, times_per_example):
    '''
    Reshape each tensor in place to have first two dimensions [B, num_temporal_splits]

    all values in output_dict must be either tf.Tensor or lists of tf.Tensor
    '''
    def _stitch_subsequences(tensor):
        shape = tensor.shape.as_list()
        assert shape[1] == times_per_example, "Tensor %s doesn't have a time dimension" % tensor.name
        true_bs = shape[0] // num_temporal_splits
        return tf.reshape(tensor, [true_bs, -1] + shape[2:])

    for k,v in output_dict.items():
        if type(v) == tf.Tensor:
            v = _stitch_subsequences(v)
            output_dict[k] = v
        elif type(v) == list:
            assert type(v[0]) == tf.Tensor
            v = [_stitch_subsequences(tensor) for tensor in v]
            output_dict[k] = v
        elif v is None:
            pass
        else:
            raise TypeError("All values in output_dict must be tf.Tensor or lists of tf.Tensors")

def collect_and_flatten(inputs, outputs, targets, **kwargs):
    print("targets in collect and flatten", targets)
    target_outputs = {}
    for target in targets:
        if target in outputs.keys():
            if isinstance(outputs[target], dict):
                for t in outputs[target]:
                    target_outputs[t] = outputs[target][t]
            else:
                target_outputs[target] = outputs[target]
    return target_outputs
