from __future__ import division, print_function, absolute_import
from collections import OrderedDict
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import copy

from tfutils import base
from tfutils.model_tool_old import conv, depth_conv
from tnn import main as tnn_main
from tnn.cell import memory as memory_func
import json
sys.path.append('../rnn_cells')
from resnetrnn import tnn_ReciprocalGateCell
from particlernn import tnn_IntegratedGraphCell, IntegratedGraphCell
from dynamics import PhysicsModel
from vectorizing import DimensionDict, aggregate_mean_and_var_features, compute_attr_spatial_moments, compute_border_attributes, add_history_attributes
from ops import mlp

import tf_nndistance
sys.path.append('../data')
# from new_data import SequenceNewDataProvider
from misc_utils import *
from rendering_utils import *
from geometry_utils import *
from training_utils import *
from model_utils import *
import shape_coding
from integrated import *
import linking
import losses as loss_functions

import pickle
from e2e_model import BuildHRNInputs

import pdb

def _get_nodes_and_segments(outputs, which_nodes, take_every=1):
    
    if which_nodes == 'spatial':
        nodes_here = outputs['spatial_nodes']
        segment_ids_here = outputs['segment_ids']
    elif which_nodes == 'summary':
        nodes_here = outputs['summary_nodes']
        segment_ids_here = outputs['summary_segment_ids']
    elif which_nodes == 'object':
        nodes_here = outputs['object_nodes']
        segment_ids_here = outputs['object_nodes']
    else:
        raise NotImplementedError("There are no such nodes as type %s" % which_nodes)
    
    print("getting nodes and segments")
    nodes_here = [n[:,take_every-1::take_every] for n in nodes_here]
    segment_ids_here = [s[:,take_every-1::take_every] for s in segment_ids_here]
    
    print(nodes_here[0].shape.as_list(), segment_ids_here[0].shape.as_list())

    return nodes_here, segment_ids_here

def input_temporal_preproc(inp_ims, static_mode, ntimes, seq_length, time_dilation, num_temporal_splits):
    if static_mode:
        if len(inp_ims.shape.as_list()) == 5:
            assert inp_ims.shape.as_list()[1] == 1
            inp_ims = [inp_ims[:,0]]*ntimes
        elif len(inp_ims.shape.as_list()) == 4:
            inp_ims = [inp_ims]*ntimes
        else:
            raise ValueError("If static_mode, inputs must either have shape [B,1,H,W,C] or [B,H,W,C]")
        
    else:
        inp_shape = inp_ims.shape.as_list()
        B,T = inp_shape[:2]
        assert seq_length == T
        assert ntimes == (seq_length * time_dilation) / num_temporal_splits
        assert seq_length % num_temporal_splits == 0, "Must split sequence into equal pieces"

        if num_temporal_splits > 1:
            inp_ims = tf.reshape(inp_ims, [B*num_temporal_splits, -1] + inp_shape[2:])
        if time_dilation > 1:
            inp_ims = dilate_tensor(inp_ims, dilation_factor=time_dilation, axis=1) # [B,seq_len*time_dilation,...]

        # make list of length num_temporal_splits, each [B,split_length,...]
        inp_ims = tf.split(inp_ims, num_or_size_splits=inp_ims.shape.as_list()[1], axis=1)
        inp_ims = inp_ims[:ntimes]
        inp_ims = [tf.squeeze(im, axis=[1]) for im in inp_ims]

    return inp_ims

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


def color_normalize_imnet(image):
    print("color normalizing")
    image = tf.cast(image, tf.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image

def coordinate_ims(batch_size, seq_len, imsize):
    bs = batch_size
    T = seq_len
    H,W = imsize
    ones = tf.ones([bs,H,W,1], dtype=tf.float32)
    h = tf.reshape(tf.div(tf.range(H, dtype=tf.float32), tf.cast(H-1, dtype=tf.float32) / 2.0),
                   [1,H,1,1]) * ones
    h -= 1.0
    w = tf.reshape(tf.div(tf.range(W, dtype=tf.float32), tf.cast(W-1, dtype=tf.float32) / 2.0),
                   [1,1,W,1]) * ones
    w -= 1.0
    h = tf.stack([h]*T, axis=1)
    w = tf.stack([w]*T, axis=1)
    hw_ims = tf.concat([h,w], axis=-1)
    return hw_ims

def build_convrnn_inputs(
        inputs_dict,
        inp_sequence_len,
        ntimes,
        time_dilation,
        num_temporal_splits,
        is_training,
        stem_model_func=None,
        stem_model_func_kwargs={},
        images_key='images64',
        hsv_input=False,
        xy_input=False,
        pmat_key='projection_matrix',
        depths_input=False,
        depths_key='depths64',
        depth_normalization=100.1,
        background_depth=30.0,
        near_plane=None,
        negative_depths=True,
        normals_from_depths_input=False,
        diff_x_input=False,
        diff_t_input=False,
        normals_input=False,
        unit_normals=True,
        normals_key='normals64',
        flows_input=False,
        flows_key='flows64',
        hw_input=False,
        ones_input=False,
        objects_mask_input=False,
        objects_mask_key='objects64',        
        color_normalize=False,
        color_scale=1.0,
        static_mode=False,
        **kwargs
):
    Dims = DimensionDict() # keep track of which channels are in which position
    if not color_normalize:
        inp_ims = tf.cast(inputs_dict[images_key], dtype=tf.float32)
        inp_ims /= color_scale
    elif color_normalize and (inputs_dict[images_key].dtype != tf.float32):
        inp_ims = color_normalize_imnet(inputs_dict[images_key])
    else:
        inp_ims = tf.cast(inputs_dict[images_key], dtype=tf.float32)
        
    inp_shape = inp_ims.shape.as_list()
    bs = inp_shape[0]    
    if len(inp_shape) == 4: # add time dimension
        H,W,C = inp_shape[1:]
        static_mode = True
        T = 1
    elif len(inp_shape) == 5 and static_mode:
        T,H,W,C = inp_shape[1:]
        inp_ims = inp_ims[:,0]
    else:
        T,H,W,C = inp_shape[1:]
    B = bs
    BT = B*T
    inp_ims = Dims.append_attr_to_vector('rgb', inp_ims)
    
    # read the depths, which are needed for computing xy
    if xy_input or depths_input:
        depths = inputs_dict[depths_key]
        depths = read_depths_image(depths, mask=None, new=True, normalization=depth_normalization, background_depth=background_depth)
        if near_plane is not None:
            depths = tf.maximum(depths, tf.cast(near_plane, tf.float32))
        if negative_depths:
            depths = -1.0 * depths
            
    # compute the image coordinates
    if xy_input or hw_input:
        hw_ims = coordinate_ims(bs,T,[H,W])
        if static_mode:
            hw_ims = hw_ims[:,0]
        
    if xy_input:
        pmat = inputs_dict[pmat_key]
        assert pmat.shape.as_list() == [B,T,4,4], "Pmat must have standard format from Unity but is %s" % (pmat.shape.as_list())
        focal_lengths = tf.stack([pmat[:,:,0,0], pmat[:,:,1,1]], axis=-1) # [B,T,2]
        xy_ims = hw_to_xy(hw_ims, depths, focal_lengths, negative_z=negative_depths, near_plane=near_plane)
        inp_ims = Dims.append_attr_to_vector('xy', xy_ims, inp_ims)

    if depths_input: # depths are stored as 3 channels of uint8; convert to single float32
        inp_ims = Dims.append_attr_to_vector('z', depths, inp_ims)        
        if xy_input:
            Dims.delete('xy')
            Dims.delete('z')
            Dims['positions'] = [-3,0]
            
    if hsv_input: # 3 channels
        rgb = tf.cast(inputs_dict[images_key], tf.float32) / 255.
        hsv = tf.image.rgb_to_hsv(rgb)
        inp_ims = Dims.append_attr_to_vector('hsv', hsv, inp_ims)                

    if diff_x_input: # sobel filter input
        diff_x = compute_sobel_features(inputs_dict[images_key], norm=255., normalize_range=True, to_rgb=True, eps=1e-8)
        inp_ims = Dims.append_attr_to_vector('sobel', diff_x, inp_ims)

    if normals_from_depths_input:
        assert depths_input, "Can't make normals from depths if not depths_input"
        if normals_from_depths_scale is None:
            normals_from_depths_scale = 900.0
        raise NotImplementedError("TODO: normals from depths via spatial derivatives")

    if normals_input:
        normals = inputs_dict[normals_key]
        normals = tf.cast(normals, dtype=tf.float32) / 255.0
        if unit_normals:
            normals = normals*2.0 - 1.0
        inp_ims = Dims.append_attr_to_vector('normals', normals, inp_ims)

    if flows_input:
        flows = inputs_dict[flows_key]
        flows = tf.image.rgb_to_hsv(tf.cast(flows, tf.float32) / 255.)
        inp_ims = Dims.append_attr_to_vector('flows', flows, inp_ims)

    if diff_t_input: # forward Euler Ims_{t+1} - Ims_{t}
        diff_t = image_time_derivative(inp_ims)
        inp_ims = tf.concat([inp_ims, diff_t], axis=-1)
        Dims.insert_from(Dims.copy(suffix='_backward_euler'))

    if hw_input:
        inp_ims = Dims.append_attr_to_vector('hw', hw_ims, inp_ims)

    if ones_input:
        ones = tf.ones_like(inp_ims[...,0:1])
        inp_ims = Dims.append_attr_to_vector('visible', ones, inp_ims)

    if objects_mask_input:
        seg_mask = tf.cast(read_background_segmentation_mask_new(inputs_dict[objects_mask_key]), dtype=tf.float32)
        inp_ims = Dims.append_attr_to_vector('foreground', seg_mask, inp_ims)

    # apply a stem function, which may reduce the imput size by a lot
    if stem_model_func is not None:
        inp_ims = stem_model_func(inp_ims, is_training=is_training, **stem_model_func_kwargs)
        
    # trim and dilate temporal input sequence
    if (inp_sequence_len is not None) and not static_mode:
        assert inp_sequence_len <= T
        inp_ims = inp_ims[:, :inp_sequence_len]
        T = inp_sequence_len

    inp_ims_list = input_temporal_preproc(inp_ims=inp_ims, static_mode=static_mode, ntimes=ntimes, seq_length=T, time_dilation=time_dilation, num_temporal_splits=num_temporal_splits)

    print("input channels")
    Dims.sort()
    for k,v in Dims.items():
        print(k,v[:2])
    print(Dims.ndims, "inp_ims", inp_ims.shape.as_list())
    
    return inp_ims_list, Dims # list in temporal order for input to TNN model

def image_time_derivative(ims):
    shape = ims.shape.as_list()
    if shape[1] == 1:
        return tf.zeros(shape, dtype=tf.float32)
    else:
        shape[1] = 1
        diff_t = ims[:,1:] - ims[:,:-1]
        diff_t = tf.concat([tf.zeros(shape=shape, dtype=tf.float32), diff_t], axis=1)
        return diff_t

def single_timestep_decoder(logits_tensor, t=-1):
    '''
    take the last time step of a a [B,T,...] tensor and squeeze
    '''
    return logits_tensor[:,t,...]

def weighted_confidence_decoder(logits_tensor, beta_range=[0.0, 50.0], beta_init=10.0):
    '''
    weight each time point by the confidence

    logits_tensor: [B,T,C]
    '''
    probs = tf.nn.softmax(logits_tensor, axis=2)
    max_probs = tf.reduce_max(probs, axis=2, keepdims=True) # [B,T,1] of max probs

    with tf.variable_scope("temporal_decoder"):
        beta = tf.get_variable("softmax_temperature",
                               shape=[], dtype=tf.float32,
                               initializer=tf.constant_initializer(beta_init))
        beta = tf.minimum(tf.maximum(beta, beta_range[0]), beta_range[1])
        time_weights = tf.nn.softmax(beta*max_probs, axis=1) # [B,T,1] sum to 1.0
        time_argmax = tf.cast(tf.argmax(time_weights[...,0], axis=1), tf.float32)
        # time_weights = tf.Print(time_weights, [tf.reduce_max(time_weights), tf.reduce_min(time_argmax), tf.reduce_mean(time_argmax), tf.reduce_max(time_argmax), beta], message="max_weight_argmax_and_beta")
        weighted_logits = tf.reduce_sum(logits_tensor * time_weights, axis=1, keepdims=False) # [B,1000]
        
    return weighted_logits

def convnet_stem(images, is_training, ksize, strides=2, hidden_ksizes=[3], hidden_separables=[False], hidden_channels=[32], out_channels=16, max_pool=True,
                 conv_kwargs={"activation": "swish"}):
    '''
    Performs the first few convolutions/nonlinearities typical of a convnet. 
    These operations also usually spatially downsample the inputs by a factor of 2-4 with strides and/or pooling
    '''
    if len(images.shape.as_list()) == 4:
        B,H,W,C = images.shape.as_list()
        T = 0
    elif len(images.shape.as_list()) == 5:
        # apply same conv to all inputs in the time dimension
        B,T,H,W,C = images.shape.as_list()
        images = tf.reshape(images, [B*T,H,W,C])
        
    if hidden_channels is None:
        hidden_channels = []
    channels = hidden_channels + [out_channels]

    print(images.name, images.shape.as_list())
    with tf.variable_scope("convnet_stem"):
        # initial conv-bn-relu for example
        with tf.variable_scope("conv0_0"):
            images = conv(images,
                          out_depth=channels[0],
                          ksize=ksize, strides=strides, padding="SAME", is_training=is_training,
                          **conv_kwargs)
            print(images.name, images.shape.as_list())
        # any extra convs
        for L, ocs in enumerate(hidden_channels):
            with tf.variable_scope("conv0_"+str(L+1)):
                conv_op = depth_conv if hidden_separables[L] else conv
                if hidden_separables[L]:
                    assert channels[L+1] == images.shape.as_list()[-1], "depth_conv can't change # of channels"
                images = conv_op(images,
                                 out_depth=channels[L+1],
                                 ksize=hidden_ksizes[L], strides=1, padding="SAME", is_training=is_training,
                                 **conv_kwargs)
                print(images.name, images.shape.as_list())                

        if max_pool:
            images = tf.nn.max_pool(images, ksize=[1,strides,strides,1], strides=[1,strides,strides,1],
                                    padding="SAME", data_format='NHWC')

    # reshape if there was a time dimension
    if T:
        images = tf.reshape(images, [B,T,H,W,C])

    return images

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
    print("border attrs", border_attrs)
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

    # XXX
    # outputs['dynamics_loss'] = {'dynamics_loss': tf.reduce_mean(tf.square(nodes) * inview_attrs)}
    
    return outputs, params

def integrated_graph_convrnn(
        inputs,
        train_targets,
        train=True,
        inp_sequence_len=4,
        num_temporal_splits=1,
        ntimes=12,
        output_times=None,
        static_mode=False,
        time_dilation=1,
        imnet_decoder=None,
        imnet_decoder_kwargs={},
        imnet_layer='imnetds',
        actions_keys=['projection_matrix'],
        target_layers=[],
        use_batch_norm=False,
        use_group_norm=False,
        num_groups=32,
        dropout_rate=0,
        memory_cell_params={},
        encoder_loss_func=None,
        encoder_loss_func_kwargs={},
        agent_loss_func=None,
        agent_loss_kwargs={},
        dynamics_model=None,
        camera_model_kwargs=None,
        dynamics_model_kwargs=None,
        dynamics_loss_func=loss_functions.hungarian_loss,
        dynamics_loss_kwargs={},
        feature_loss_func=None,
        feature_loss_func_kwargs={},
        selfsup_loss_func=None,
        selfsup_loss_func_kwargs={},
        rendering_loss_func_kwargs=None,
        spatial_attributes_to_decode=None,
        spatial_decoder_kwargs={},
        future_attributes_to_decode=None,
        future_decoder_kwargs={},
        relative_attr_loss_kwargs=None,
        edges_error_signal=None,
        output_attrs=False,
        output_hierarchy=False,
        output_global_pool=False,
        get_segment_ids=False,
        hrn_scope=None,
        hrn_nodes='summary',
        hier_kwargs={},
        hrn_func=None,
        hrn_kwargs=None,
        hrn_loss_func=None,
        hrn_loss_kwargs=None,
        use_gt_next_velocity_supervision=True,
        **model_kwargs
        ):

    params = {}
    outputs = {}
    labels = []

    inputs_temporal_list, cdict = build_convrnn_inputs(inputs, inp_sequence_len=inp_sequence_len,
                                                       ntimes=ntimes, time_dilation=time_dilation,
                                                       num_temporal_splits=num_temporal_splits,
                                                       is_training=train, **model_kwargs)

    inputs_temporal_list = [tf.identity(inp, name=(model_kwargs.get('input_name', 'input_t')+str(t))) for t,inp in enumerate(inputs_temporal_list)]

    # build actions: the camera or projection matrices
    actions_inputs = []
    if actions_keys is None:
        actions_keys = []
    for akey in actions_keys:
        action = inputs[akey]
        # action = tf.reshape(action, action.shape.as_list()[:2] + [-1])
        action_temporal_list = input_temporal_preproc(action, static_mode=static_mode, ntimes=ntimes,
                                                      seq_length=inp_sequence_len, time_dilation=time_dilation,
                                                      num_temporal_splits=num_temporal_splits)
        actions_inputs.append(action_temporal_list)
        print(akey, action_temporal_list)
    
    batch_size = inputs_temporal_list[0].shape.as_list()[0] # now B*num_temporal_splits
    inp_shape = inputs_temporal_list[0].shape.as_list()[1:]
    print("len inputs", len(inputs_temporal_list))
    print("full inp shape", inputs_temporal_list[0].shape.as_list())
    print("bs and inp-shape", batch_size, inp_shape)

    if not train:
        outputs['image_inputs'] = tf.stack([inputs_temporal_list[t] for t in output_times], axis=1) # [B*nts,len(output_times)]+inp_shape

    # TNN model with both integrated ConvRNN/Graph cells
    with tf.variable_scope('tnn_model'):
        G_tnn = tnn_main.graph_from_json(model_kwargs['base_name'] +'.json') # builds a NetworkX graph of the model

        # loop over nodes (i.e. layers in the TNN model)
        for node, attr in G_tnn.nodes(data=True):

            # set input layer shape correctly
            if node == model_kwargs.get('input_layer', 'conv1'):
                attr['shape'] = inp_shape
            
            # set batchnorm, dropout in pre_ and post_memory
            for func, kwargs in attr['kwargs']['pre_memory'] + attr['kwargs']['post_memory']:
                if 'batch_norm' in kwargs:
                    kwargs['batch_norm'] = use_batch_norm and (not use_group_norm)
                    kwargs['group_norm'] = use_group_norm
                    kwargs['num_groups'] = num_groups
                if kwargs.get('batch_norm', False):
                    kwargs['is_training'] = train
                if func.__name__ == 'fc': # dropout applies to all fc functions in all nodes
                    kwargs['dropout'] = dropout_rate if (dropout_rate and train) else None

            # set memory params in each layer (overwrite JSON)
            layer_memory_cell_params = copy.deepcopy(memory_cell_params.get(node, {}))
            if len(layer_memory_cell_params):
                layer_memory_cell_params['is_training'] = train                                            
                attr['cell'] = tnn_IntegratedGraphCell
                attr['kwargs']['memory'] = (memory_func, layer_memory_cell_params)
            else:
                attr['kwargs']['memory'] = (memory_func, {})

            # get rid of an imnet decoder
            if 'imnet' in node and output_global_pool:
                # find the global pool outputs and chop off everything after
                global_pool = False
                for i, func in enumerate(attr['kwargs']['pre_memory']):
                    if func[0].__name__ == 'global_pool':
                        global_pool = True
                    elif global_pool:
                        attr['kwargs']['pre_memory'][i] = (tf.identity, {})

            # if no decoder, then get rid of layer
            elif 'imnet' in node and (imnet_decoder is None):
                G_tnn.remove_node(node)
                
        print("nodes", [n for n,_ in G_tnn.nodes(data=True)])
        # add nontrivial model edges (skip, feedback)
        edges = model_kwargs.get('edges', [])
        if len(edges):
            print("applying edges", edges)
            G_tnn.add_edges_from(edges)

    # Initialize TNN model
    tnn_input_layers = model_kwargs.get('input_layer', 'conv1')
    if not isinstance(tnn_input_layers, list):
        tnn_input_layers = [tnn_input_layers]
    tnn_inputs = {tnn_input_layers[0]: inputs_temporal_list}
    for a, akey in enumerate(actions_keys):
        tnn_inputs[tnn_input_layers[1+a]] = actions_inputs[a]

    motion_input_layer = model_kwargs.get('motion_input_layer', None)
    if motion_input_layer is not None:
        if motion_input_layer not in tnn_input_layers:
            tnn_input_layers.append(motion_input_layer)
        delta_ims = image_time_derivative(tf.cast(inputs['images'], tf.float32) / 255.)
        delta_ims_list = input_temporal_preproc(inp_ims=delta_ims, static_mode=static_mode, ntimes=ntimes, seq_length=inp_sequence_len, time_dilation=time_dilation, num_temporal_splits=num_temporal_splits)
        delta_ims_list = [tf.identity(dim, name='motion_split'+str(t)) for t,dim in enumerate(delta_ims_list)]
        tnn_inputs[motion_input_layer] = delta_ims_list

    tnn_main.init_nodes(G_tnn, input_nodes=tnn_input_layers, batch_size=batch_size, channel_op='concat')        
        
    # Unroll TNN model
    if model_kwargs.get('unroll_tf', True):
        tnn_main.unroll_tf(G_tnn, input_seq=tnn_inputs, ntimes=ntimes, ff_order=model_kwargs.get('ff_order', None))
    else: # tnn unroller, "biological"
        tnn_main.unroll(G_tnn, input_seq=tnn_inputs, ntimes=ntimes)


    # Get conv2d and/or graph node outputs
    layers_with_graph_nodes = sorted([L for L, layer_params in memory_cell_params.items()
                                      if layer_params.get('graph_cell_kwargs', None) is not None])


    spatial_nodes = []
    summary_nodes = []
    object_nodes = []
    spatial_to_summary_ids = []
    summary_to_object_ids = []
    segment_edges = []
    n2p_edges = []
    ws_edges = []
    num_nodes = []
    segment_ids = []
    edges = []
    errors = []
    time_errors = []
    space_errors = []
    dyn_errors = []
    summary_segment_ids = []
    static_segment_ids = []
    object_segment_ids = []
    motion_segment_ids = []
    tracking_inds = []
    output_times = range(ntimes) if output_times is None else output_times
    get_edges = model_kwargs.get('get_edges', False)
    vae_loss = []
    dynamic_edge_loss = []
    
    # global pool outputs for brainscore metrics
    if output_global_pool:
        outputs['times'] = {t: tf.squeeze(G_tnn.node[imnet_layer]['outputs'][t]) for t in output_times}
        print("global pol outputs", outputs)
        return outputs, params

    # graph outputs
    for L, layer in enumerate(layers_with_graph_nodes):
        layer_graph_cell = [G_tnn.node[layer]['states'][t]['graph_cell_state']
                            for t in output_times] # list of length ntimes; each entry is state dict
        
        spatial_nodes.append(tf.stack([gc['spatial_nodes'] for gc in layer_graph_cell], axis=1))
        summary_nodes.append(tf.stack([gc['summary_nodes'] for gc in layer_graph_cell], axis=1))
        object_nodes.append(tf.stack([gc['object_nodes'] for gc in layer_graph_cell], axis=1))        
        spatial_to_summary_ids.append(tf.stack([gc['spatial_to_summary_ids'] for gc in layer_graph_cell], axis=1))
        summary_to_object_ids.append(tf.stack([gc['summary_to_object_ids'] for gc in layer_graph_cell], axis=1))                
        segment_edges.append(tf.stack([gc['segment_edges'] for gc in layer_graph_cell], axis=1))
        tracking_inds.append(tf.stack([gc['tracking_inds'] for gc in layer_graph_cell], axis=1))

        if model_kwargs.get('edge_vae_loss', False):
            # vae_loss += tf.reduce_mean(tf.add_n([gc.get('vae_loss', tf.constant([0.0])) for gc in layer_graph_cell]))
            vae_loss.append(tf.stack([gc.get('vae_loss', tf.constant([0.0])) for gc in layer_graph_cell], axis=1))
            dynamic_edge_loss.append(tf.stack([gc.get('dynamic_edge_loss', tf.constant([0.0])) for gc in layer_graph_cell], axis=1))            
            edges.append(tf.stack([gc.get('edges', tf.zeros([batch_size])) for gc in layer_graph_cell], axis=1))
            errors.append(tf.stack([gc.get('errors', tf.zeros([batch_size])) for gc in layer_graph_cell], axis=1))
            time_errors.append(tf.stack([gc.get('time_errors', tf.zeros([batch_size])) for gc in layer_graph_cell], axis=1))
            space_errors.append(tf.stack([gc.get('space_errors', tf.zeros([batch_size])) for gc in layer_graph_cell], axis=1))
            dyn_errors.append(tf.stack([gc.get('dyn_errors', tf.zeros([batch_size])) for gc in layer_graph_cell], axis=1))
            
        if get_segment_ids:
            layer_shape = layer_graph_cell[0]['spatial_output'].shape.as_list()[1:3]
            segment_ids.append(tf.concat([tf.reshape(gc['segment_ids'], [batch_size,1, layer_shape[0], layer_shape[1]]) for gc in layer_graph_cell], axis=1))
            summary_segment_ids.append(tf.concat([tf.reshape(gc['summary_segment_ids'], [batch_size,1, layer_shape[0], layer_shape[1]]) for gc in layer_graph_cell], axis=1))
            object_segment_ids.append(tf.concat([tf.reshape(gc['object_segment_ids'], [batch_size, 1, layer_shape[0], layer_shape[1]]) for gc in layer_graph_cell], axis=1))
            static_segment_ids.append(tf.concat([tf.reshape(gc['static_segment_ids'], [batch_size, 1, layer_shape[0], layer_shape[1]]) for gc in layer_graph_cell], axis=1))
            motion_segment_ids.append(tf.concat([tf.reshape(gc['motion_segment_ids'], [batch_size, 1, layer_shape[0], layer_shape[1]]) for gc in layer_graph_cell], axis=1))                        
            

        if get_edges:
            if len(layer_graph_cell[-1]['summary_adjacency_matrix']):
                n2p_edges.append(tf.concat([gc['summary_adjacency_matrix'][0]['n2p_edges'] for gc in layer_graph_cell[1:]], axis=1))
                ws_edges.append(tf.concat([gc['summary_adjacency_matrix'][0]['ws_edges'] for gc in layer_graph_cell[1:]], axis=1))
            else:
                n2p_edges.append(None)
                ws_edges.append(None)
    
    outputs.update({
        'segment_edges': segment_edges,
        'spatial_nodes': spatial_nodes,
        'summary_nodes': summary_nodes,
        'object_nodes': object_nodes,
        'spatial_to_summary_ids': spatial_to_summary_ids,
        'summary_to_object_ids': summary_to_object_ids,
        'segment_ids': segment_ids if get_segment_ids else None,
        'summary_segment_ids': summary_segment_ids if get_segment_ids else None,
        'object_segment_ids': object_segment_ids if get_segment_ids else None,
        'static_segment_ids': static_segment_ids if get_segment_ids else None,
        'motion_segment_ids': motion_segment_ids if get_segment_ids else None,        
        'tracking_inds': tracking_inds,
        'edges': edges,
        'errors': errors,
        'time_errors': time_errors,
        'space_errors': space_errors,
        'dyn_errors': dyn_errors,
        'n2p_edges': n2p_edges if get_edges else None,
        'ws_edges': ws_edges if get_edges else None,
        'vae_loss': vae_loss if model_kwargs.get('edge_vae_loss', False) else None,
        'dynamic_edge_loss': dynamic_edge_loss if model_kwargs.get('edge_vae_loss', False) else None
        # 'vae_loss': {'vae_loss': vae_loss} if model_kwargs.get('edge_vae_loss', False) else None        
    })

    # conv2d features outputs
    for layer in target_layers:
        targ_key = model_kwargs.get('target_key', 'outputs')
        if targ_key == 'outputs':
            outputs[layer] = tf.stack([G_tnn.node[layer][targ_key][t] for t in output_times], axis=1)
        else:
            outputs[layer] = tf.stack([G_tnn.node[layer]['states'][t][targ_key] for t in output_times], axis=1)

    # stitch the input subsequences back together
    if num_temporal_splits > 1:
        batch_size = batch_size // num_temporal_splits
        combine_temporal_splits(outputs, num_temporal_splits, times_per_example=len(output_times))
    take_every = len(output_times)
    output_times = range(len(output_times)*num_temporal_splits)

    print("outputs")
    for k,out in outputs.items():
        print(k, type(out))
        if type(out) == tf.Tensor:
            print(out.shape.as_list())
        elif type(out) == list:
            print("len %d" % len(out))
            print(out[0].shape.as_list())
            print(" ")

    print("take every, output_times", take_every, output_times)
    
    # for base.train_from_params
    if outputs['vae_loss'] is not None:
        vae_loss = outputs['vae_loss']
        outputs['vae_loss'] = {'vae_loss':
                               tf.add_n(
                                   [tf.reduce_mean(vloss, axis=[0,1])
                                   for vloss in vae_loss])}
    if outputs['dynamic_edge_loss'] is not None:
        de_loss = outputs['dynamic_edge_loss']
        outputs['dynamic_edge_loss'] = {'dynamic_edge_loss':
                               tf.add_n(
                                   [tf.reduce_mean(dloss, axis=[0,1])
                                   for dloss in de_loss])}
    else:
        outputs['dynamic_edge_loss'] = tf.cast(0.0, tf.float32)


    # agent model and loss: predict what happens when the agent does something
    if agent_loss_func is not None and train:
        which_nodes = model_kwargs.get('agent_nodes', 'spatial')
        agent_tiers = model_kwargs.get('agent_tiers', [0])
        if not isinstance(which_nodes, list):
            which_nodes = [which_nodes]
        agent_loss = tf.cast(0.0, tf.float32)
        for tier in agent_tiers:
            for level, node_level in enumerate(which_nodes):
                nodes_here, segments_here = _get_nodes_and_segments(outputs, node_level, take_every)
                print("agent_nodes", nodes_here[tier].shape.as_list())

                # todo: transform nodes according to actions

                # get loss from correspondence
                agent_loss_args = [inputs['projection_matrix'], inputs['camera_matrix']]
                if isinstance(agent_loss_kwargs, dict):
                    agent_loss_kwargs_level = copy.deepcopy(agent_loss_kwargs)
                elif isinstance(agent_loss_kwargs, list):
                    agent_loss_kwargs_level = agent_loss_kwargs[level]
                # agent loss func
                agent_loss += agent_loss_func(nodes_here[tier], *agent_loss_args, **agent_loss_kwargs_level)
        outputs['agent_loss'] = {'agent_loss': agent_loss}

    # dynamics model and loss; compare pushed forward nodes with subsequent observations
    if (dynamics_model is not None) or (camera_model_kwargs is not None) or (model_kwargs.get('segmentation_loss_scale', 0)):
        which_nodes = model_kwargs.get('dynamics_nodes', 'summary')
        nodes_here, segments_here = _get_nodes_and_segments(outputs, which_nodes, take_every) # len(tiers) list of [B,T,N,Dtier] nodes
        d_tier = model_kwargs.get('dynamics_tier', 0)
        nodes_here = nodes_here[d_tier] # get the first tier
        segments_here = segments_here[d_tier]
        print("dynamics nodes_here", nodes_here.shape.as_list())        
        nodes_trans_dict = model_kwargs.get('nodes_trans_dict', OrderedDict({'spatial_nodes': 'nodes_level_0', 'summary_nodes': 'nodes_level_1'}))
        dynamics_loss = tf.cast(0.0, tf.float32)

        # permute the nodes based on predicted matches to get full trajectories
        tstart = model_kwargs.get('start_predictions_time', 1)
        if dynamics_loss_kwargs is not None:
            num_loss_dims = len(dynamics_loss_kwargs.get('loss_weights', [1.0]))
        else:
            num_loss_dims = 1
        node_trajectories = [tf.concat([nodes_here[:,tstart-1], tf.zeros_like(nodes_here[:,tstart-1,:,-num_loss_dims:])], axis=-1)] # add extra dim for loss
        match = tf.tile(tf.range(node_trajectories[0].shape.as_list()[1], dtype=tf.int32)[tf.newaxis,:], [batch_size,1]) # init to range(N)
        pred_graphs = []

        # get loss from comparing forward predictions to observations; last next nodes cant be compared to anything
        num_times = nodes_here.shape.as_list()[1]
        next_nodes = []
        aligned_next_nodes = []
        rollout_type = model_kwargs.get('rollout_type', 'all')
        assert rollout_type in ['all', 'first', 'last']
        
        # build model
        if dynamics_model:
            dynamics_model = PhysicsModel(**dynamics_model_kwargs)
        else:
            tstart = num_times
            rollout_type = 'none'

        for t in range(tstart, num_times):
            print("num input times now", t, "up to", num_times-1)
            # get prev nodes 0,...,t-1
            prev_nodes_t = {k: outputs[k][d_tier][:,:t] # [B,t,N,D]
                            for k in nodes_trans_dict.keys()}
            edges_t = outputs['spatial_to_summary_ids'][d_tier][:,:t] # [B,t,N]

            # transform agent-relative coordinates
            if camera_model_kwargs is not None:
                cmats = inputs['camera_matrix'][:,0:t+1] # [B,t+1,4,4]
                prev_nodes_t = {k:camera_rigid_motion_model(prev_nodes_t[k], cmats, **camera_model_kwargs)
                                for k in nodes_trans_dict.keys()}

            # build the dynamics model and run it forward all possible times
            pmat = inputs['projection_matrix'][:,:t]

            if rollout_type == 'all':
                rollouts = range(t, num_times)
            elif rollout_type == 'first':
                rollouts = range(t, t+1)
            elif rollout_type == 'last':
                rollouts = range(num_times-1, num_times)
            elif rollout_type == 'none':
                rollouts = []
            # for tfin in range(t, num_times):
            for tfin in rollouts:
                num_rollout_times = tfin - t + 1
                next_nodes_t = dynamics_model.predict(
                    input_nodes=[prev_nodes_t[k] for k in nodes_trans_dict.keys()],
                    input_edges=[edges_t],
                    projection_matrix=pmat,
                    num_rollout_times=num_rollout_times,
                    output_node_keys=nodes_trans_dict.values(),
                    **dynamics_model_kwargs) 
                next_nodes_t = next_nodes_t[nodes_trans_dict[which_nodes+'_nodes']] # the output nodes [B,num_rollout_times,N,D]                
                print("inp_t", t, "pred_t", t-1+next_nodes_t.shape.as_list()[1], "tfin", tfin, "num_rollout_times", num_rollout_times, "preds", next_nodes_t.shape.as_list())
                # append the predicted next nodes
                if num_rollout_times == 1 and rollout_type != 'last':
                    next_nodes.append(next_nodes_t[:,0])
                    if not train:
                        pred_graphs += dynamics_model.G_pred[0:1]
                elif rollout_type == 'last':
                    next_nodes.append(next_nodes_t[:,0])
                    
                if dynamics_loss_func is not None:
                    if model_kwargs.get('match_with_init_nodes', False):
                        nodes_init_t = prev_nodes_t[which_nodes+'_nodes'][:,-1]
                    else:
                        nodes_init_t = None
                    loss_tfin, nodes_tfin_matched, match_t = dynamics_loss_func( # pred, gt, [init], **kwargs
                        next_nodes_t[:,-1], nodes_here[:,tfin], nodes_init=nodes_init_t, **dynamics_loss_kwargs)
                    print("loss, nodes_matched, match", loss_tfin.shape.as_list(), nodes_tfin_matched.shape.as_list(), match_t.shape.as_list())
                    dynamics_loss += loss_tfin
                    if num_rollout_times == 1:
                        node_trajectories.append(permute_nodes(nodes_tfin_matched, match))
                        match = permute_nodes(match_t[...,tf.newaxis], match)[...,0]                        

        # stitch together nodes
        node_trajectories = tf.stack(node_trajectories, axis=1)

        # future images loss
        if num_times > 1 and len(next_nodes):
            next_nodes = tf.stack(next_nodes, axis=1) # [B,T-1,N,D]

        # add a cross-entropy loss on the segmentation mask
        shapes_loss = tf.cast(0.0, tf.float32)
        if model_kwargs.get('segmentation_loss_scale', 0):
            scale = model_kwargs['segmentation_loss_scale']
            size = segments_here.shape.as_list()[-2:]            
            _, segment_logits, _, _ = shape_images_from_nodes_decoder(
                nodes_here, # [B,T,N,D]
                imsize=size,
                hard_segments=False,
                attributes_to_decode=[],
                attribute_dims=None,
                attribute_postprocs=None,
                **future_decoder_kwargs)

            segment_logits = tf.transpose(segment_logits, [0,1,3,4,2]) # [B,T,H,W,N]
            Nmax = segment_logits.shape.as_list()[-1]
            segment_labels, valid_segments, num_valid_px = preproc_segment_ids(segments_here, Nmax=Nmax, return_valid_segments=True)
            seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=mask_tensor(segment_logits, valid_segments[...,tf.newaxis], mask_value=(1./Nmax)),
                labels=segment_labels) # [B,T,H,W]

            seg_loss = tf.reduce_mean(tf.reduce_sum(seg_loss * valid_segments, axis=[2,3]) / tf.maximum(1.0, num_valid_px))
            # seg_loss = tf.Print(seg_loss, [seg_loss], message='seg_loss')
            shapes_loss += scale * seg_loss

        # future images
        tstart = model_kwargs.get('start_rollout_time', tstart)
        future_pred_images = future_segments = future_nodes = shapes = shape_codes = None
        if model_kwargs.get('future_image_loss_scale', 0) and future_attributes_to_decode is not None and (tstart < num_times):
            
            # get the labels
            future_gt_images = [inputs[k][:,tstart:inp_sequence_len] for k in model_kwargs['future_labels_keys']]
            future_gt_images = [func(future_gt_images[i]) for i,func in enumerate(model_kwargs['future_labels_preprocs'])]
            size = segments_here.shape.as_list()[-2:]            
            future_gt_images = [
                tf.reshape(tf.image.resize_images(
                    tf.reshape(im, [-1]+im.shape.as_list()[2:]), size), im.shape.as_list()[:2] + size + im.shape.as_list()[-1:])
                for im in future_gt_images]
            
            if 'valid' in inputs:
                future_valid_images = tf.cast(inputs['valid'][:,tstart:inp_sequence_len], tf.float32)
                future_valid_images = tf.reshape(
                    tf.image.resize_images(
                        tf.reshape(future_valid_images, [-1]+future_valid_images.shape.as_list()[2:]), size),
                    future_valid_images.shape.as_list()[:2] + size + [1])
            else:
                future_valid_images = tf.ones_like(future_gt_images[0][...,0:1])
            future_valid_images = [future_valid_images if val else tf.ones_like(future_valid_images)
                                   for val in future_decoder_kwargs.get('mask_invalid', [False]*len(future_gt_images))]

            future_nodes = next_nodes if model_kwargs.get('decode_forward_nodes', False) else nodes_here[:,tstart:]
            future_pred_images, future_segments, shapes, shape_codes = shape_images_from_nodes_decoder(
                future_nodes,
                imsize=size,
                hard_segments=True,
                attributes_to_decode=future_attributes_to_decode,
                attribute_dims=hier_kwargs['node_attribute_dims'][d_tier],
                attribute_postprocs=hier_kwargs['node_attribute_preprocs'][d_tier],
                **future_decoder_kwargs)

            if model_kwargs.get('future_motion_sampling', False):
                assert num_times > tstart+1, "must have at least two times"
                im_size = inputs['images'].shape.as_list()[-3:-1]
                strides = tf.reshape(tf.constant([im_size[0] // size[0], im_size[1] // size[1]], dtype=tf.int32), [1,1,1,2])
                motion_inds = sample_delta_image_inds(inputs['images'][:,:inp_sequence_len], num_points=future_decoder_kwargs['num_motion_points'])
                if model_kwargs.get('future_static_sampling', False):
                    static_inds = sample_delta_image_inds(inputs['images'][:,:inp_sequence_len], num_points=future_decoder_kwargs['num_static_points'], static=True)
                    motion_inds = tf.concat([static_inds, motion_inds], axis=2)
                motion_inds = tf.floordiv(motion_inds, strides)
                for j in range(len(future_pred_images)):
                    scale = future_decoder_kwargs['loss_scales'][j] * model_kwargs['future_image_loss_scale']
                    pred_values = get_image_values_from_indices(future_pred_images[j], motion_inds)
                    gt_values = get_image_values_from_indices(future_gt_images[j], motion_inds)
                    fut_loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred_values - gt_values), axis=-1))
                    grads = tf.gradients(fut_loss, shape_codes)
                    # fut_loss = tf.Print(fut_loss, [tf.reduce_mean(tf.abs(grads))], message='shape_code_grads_'+str(j))
                    # fut_loss = tf.Print(fut_loss, [tf.reduce_min(shape_codes, axis=[0,1,2,3]),
                    #                                tf.reduce_max(shape_codes, axis=[0,1,2,3])], message='minmax_shape_codes_'+str(j))                        
                    # fut_loss = tf.Print(fut_loss, [fut_loss], message='fut_loss_attr_'+str(j))
                    shapes_loss += scale * fut_loss

            else:
                for j,pred_im in enumerate(future_pred_images):
                    scale = tf.constant(future_decoder_kwargs['loss_scales'][j] * model_kwargs['future_image_loss_scale'], tf.float32)
                    # future_gt_images[j] = tf.Print(future_gt_images[j], [tf.reduce_max(tf.abs(pred_im)),
                    #                                                      tf.reduce_max(tf.abs(future_gt_images[j]))], message='max_fut_im_'+str(j))
                    fut_loss = tf.reduce_mean(tf.reduce_sum(future_valid_images[j] * tf.square(pred_im - future_gt_images[j]), axis=-1))

                    # fut_loss = tf.Print(fut_loss, [fut_loss, scale], message='fut_loss_attr_'+str(j))
                    shapes_loss += scale * fut_loss

            # rollouts beyond the training images
            if (not train) and model_kwargs.get('future_rollout_times', 0):
                assert isinstance(dynamics_model, PhysicsModel)
                tstart = model_kwargs.get('start_rollout_time', num_times-1)
                print("tstart", tstart)
                num_future_times = model_kwargs['future_rollout_times']
                next_nodes_t = dynamics_model.predict(
                    input_nodes=[v for v in {k: outputs[k][d_tier][:,:tstart+1] for k in nodes_trans_dict.keys()}.values()],
                    input_edges=[outputs['spatial_to_summary_ids'][d_tier][:,:tstart+1]],
                    projection_matrix=pmat[:,:tstart+1],
                    num_rollout_times=num_future_times,
                    output_node_keys=nodes_trans_dict.values(),
                    **dynamics_model_kwargs)
                next_nodes_t = next_nodes_t[nodes_trans_dict[which_nodes+'_nodes']] # the output nodes [B,num_rollout_times,N,D]
                next_images, next_segments, next_shapes, next_shape_codes = shape_images_from_nodes_decoder(
                    next_nodes_t,
                    imsize=size,
                    hard_segments=True,
                    attributes_to_decode=future_attributes_to_decode,
                    attribute_dims=hier_kwargs['node_attribute_dims'][d_tier],
                    attribute_postprocs=hier_kwargs['node_attribute_preprocs'][d_tier],
                    **future_decoder_kwargs)

                # add to predictions
                for i,next_im in enumerate(next_images):
                    future_pred_images[i] = tf.concat([future_pred_images[i], next_im], axis=1)
                future_segments = tf.concat([future_segments, next_segments], axis=1)
                future_nodes = tf.concat([future_nodes, next_nodes_t], axis=1)
                pred_graphs += dynamics_model.G_pred
                shapes = tf.concat([shapes, next_shapes], axis=1)
                shape_codes = tf.concat([shape_codes, next_shape_codes], axis=1)
                print("future segments now", future_segments.shape.as_list())
                print("future shapes now", shapes.shape.as_list())
                print("future shape codes now", shape_codes.shape.as_list())
                print("future nodes now", future_nodes.shape.as_list())
                print("Pred Graphs", len(pred_graphs), type(pred_graphs[0]))

        # update outputs and loss
        print("node trajectories")
        print(node_trajectories)

        outputs.update({
            'node_trajectories': [node_trajectories],
            'dynamics_loss': {'dynamics_loss': dynamics_loss},
            'shapes_loss': {'shapes_loss': shapes_loss}
        })
        if future_pred_images is not None and (not train):
            outputs['future_images'] = {k:future_pred_images[i] for i,k in enumerate(model_kwargs['future_labels_keys'])}
        if future_segments is not None and (not train):
            outputs['future_segments'] = future_segments
        if future_nodes is not None and (not train):
            outputs['future_nodes'] = future_nodes
        if shapes is not None and (not train):
            outputs['shapes'] = shapes
            outputs['shape_codes'] = shape_codes
        if not train and rollout_type != 'last' and (dynamics_model_kwargs is not None):
            nk = nodes_trans_dict[nodes_trans_dict.keys()[-1]]
            outputs['pred_graph'] = {attr: tf.concat([G['nodes'][nk][attr] for G in pred_graphs], axis=1) for attr in pred_graphs[0]['nodes'][nk].keys()}

    # convert graph to attrs
    if output_attrs:
        if not isinstance(hrn_nodes, list):
            encoder_outputs = linking.get_node_attributes(outputs, which_nodes=hrn_nodes, **hier_kwargs)
            outputs['encoder_outputs'] = encoder_outputs            
        elif isinstance(hrn_nodes, list):
            # dict of lists of dicts...yikes
            encoder_outputs = {nodes: linking.get_node_attributes(outputs, which_nodes=nodes, **hier_kwargs.get(nodes, hier_kwargs))
                               for nodes in hrn_nodes}
            outputs['encoder_outputs'] = encoder_outputs[hrn_nodes[0]]
        
                
    # node supervision etc.
    if encoder_loss_func is not None:
        encoder_loss = {}
        encoder_loss['encoder_loss'] = encoder_loss_func(
                logits=outputs,
                labels=labels,
                **encoder_loss_func_kwargs)
        outputs.update({'encoder_loss': encoder_loss})

    # e.g. for feature smoothing
    if feature_loss_func is not None:
        if not isinstance(feature_loss_func, list):
            feature_loss_func = [feature_loss_func]
            feature_loss_func_kwargs = [feature_loss_func_kwargs]
        feature_loss = 0.0
        for i, func in enumerate(feature_loss_func):
            key = feature_loss_func_kwargs[i].get('spatial_labels_key', 'segment_edges')
            try:
                spatial_labels = outputs[key]
                if isinstance(spatial_labels, list):
                    spatial_labels = spatial_labels[L]
            except KeyError:
                spatial_labels = inputs[key]
            lshape = spatial_labels.shape.as_list()
            if len(lshape) == 4 and take_every > 1:
                spatial_labels = tf.reshape(
                    tf.tile(spatial_labels[:,tf.newaxis], [1,take_every,1,1,1]),
                    [-1,lshape[1:]])
            elif len(lshape) == 5:
                spatial_labels = dilate_tensor(spatial_labels, dilation_factor=take_every, axis=1)
                spatial_labels = tf.reshape(spatial_labels, [-1] + lshape[2:])

                # valid images
                if 'valid' in inputs:
                    spatial_valid = tf.reshape(
                        dilate_tensor(inputs['valid'], dilation_factor=take_every, axis=1),
                        spatial_labels.shape.as_list()[:-1] + [1])
                else:
                    spatial_valid = tf.ones(shape=spatial_labels.shape, dtype=tf.bool)

            # inner loop over layers
            for L, layer in enumerate(model_kwargs.get('feature_layers', target_layers)):
                features = outputs[layer]
                fshape = features.shape.as_list()
                features = tf.reshape(features, [-1] + fshape[2:])
                feature_loss += func(
                    features, labels=spatial_labels,
                    features_name=layer,
                    valid_images=spatial_valid,
                    **feature_loss_func_kwargs[i])
                
        outputs.update({'feature_loss': {'feature_loss':feature_loss}})

    # rendering nodes into rgb, normals, depths, etc.
    if rendering_loss_func_kwargs is not None:
        Rtiers = model_kwargs.get('render_tiers', [0])
        Rattrs = model_kwargs.get('render_attrs', ['pred_particles', 'alphas', 'pred_colors', 'pred_normals', 'pred_object_masks', 'is_moving'])
        if not isinstance(rendering_loss_func_kwargs, list):
            rendering_loss_func_kwargs_list = [rendering_loss_func_kwargs] * len(Rtiers)
        else:
            rendering_loss_func_kwargs_list = copy.deepcopy(rendering_loss_func_kwargs)
        
        # pass to loss func
        render_loss = 0.0
        for tier in Rtiers:
            if isinstance(encoder_outputs, list):
                render_logits = {k:encoder_outputs[tier][k] for k in Rattrs}
            elif isinstance(encoder_outputs, dict):
                render_logits = {k:encoder_outputs[model_kwargs.get('render_nodes', 'spatial')][tier][k] for k in Rattrs}

            for layer in target_layers:
                render_logits[layer] = outputs[layer]

            render_loss += rendering_particles_loss(
                logits=[render_logits],
                labels={k:inp[:,:inp_sequence_len] for k,inp in inputs.items()},
                **rendering_loss_func_kwargs_list[tier])
        outputs.update({'render_loss': {'render_loss':render_loss}})

    # decode spatial attributes
    if spatial_attributes_to_decode is not None:
        # todo get points to decode
        im_size = inputs[model_kwargs.get('images_key', 'images')].shape.as_list()[-3:-1]
        spatial_inds = sample_image_inds(out_shape=[batch_size, len(output_times), spatial_decoder_kwargs.get('num_decoder_points', 1024)],
                                         im_size=im_size,
                                         train=train,
                                         **spatial_decoder_kwargs)
        
        # get decoders from scene node or higher-tier spatial nodes
        decoder_tier = spatial_decoder_kwargs.get('decoder_tier', -1)
        decoder_nodes = spatial_decoder_kwargs.get('decoder_nodes', 'spatial')
        try:
            decoders = {attr: encoder_outputs[decoder_tier][attr + '_decoder'] for attr in spatial_attributes_to_decode}
        except KeyError:
            try: 
                decoders = {attr: encoder_outputs[decoder_nodes][decoder_tier][attr + '_decoder'] for attr in spatial_attributes_to_decode}
            except KeyError:
                # decoders are just model parameters, not node predictions
                decoders = None

        # if decoders[spatial_attributes_to_decode[0]].shape.as_list()[2] != 1:
        #     # todo: hierarchy where each node has a separate decoder
        #     raise NotImplementedError("use n2a indices to get the decoder for each spatial node")
        
        # get the decoded attrs
        which_nodes = model_kwargs.get('spatial_decoder_nodes', 'spatial')        
        if isinstance(encoder_outputs, list):
            attrs_here = encoder_outputs
            segment_ids_here = segment_ids
        else:
            attrs_here = encoder_outputs[which_nodes]
            if which_nodes == 'spatial':
                segment_ids_here = segment_ids
            elif which_nodes == 'summary':
                segment_ids_here = summary_segment_ids

        spatial_tiers = spatial_decoder_kwargs.get('spatial_tiers', [0])                
        decoded_attrs = []
        spatial_attrs_loss = tf.cast(0.0, tf.float32)
        # for some reason tensorflow loses track of these shapes
        segment_ids_here = [tf.reshape(
            segs, [batch_size,len(output_times)]+segs.shape.as_list()[-2:])
                            for segs in segment_ids_here]
                            
        for tier in spatial_tiers:
            valid_vectors = attrs_here[tier]['valid']            
            d_attrs, v_attrs, node_inds_per_point = spatial_attribute_decoder(
                spatial_inds=spatial_inds,
                segment_ids=segment_ids_here[tier],
                segment_centroids=attrs_here[tier]['hw_centroids'],
                vectors=attrs_here[tier]['vector'],
                valid_vectors=valid_vectors,
                decoders=decoders,
                attributes_to_decode=spatial_attributes_to_decode,
                attribute_dims=hier_kwargs['node_attribute_dims'][tier],
                attribute_postprocs=hier_kwargs['node_attribute_preprocs'][tier],
                im_size=im_size,
                **spatial_decoder_kwargs
            )
            d_attrs['alphas'] = v_attrs
            d_attrs['valid'] = v_attrs
            d_positions = attrs_here[tier].get('pred_particles', attrs_here[tier]['vector'][...,0:3])
            d_attrs['pred_particles'] = tf.gather_nd(d_positions, node_inds_per_point) # [B,T,P,3]
            decoded_attrs.append(d_attrs)

            # get spatial sampling losses
            spatial_loss_kwargs = spatial_decoder_kwargs.get('rendering_kwargs', copy.deepcopy(rendering_loss_func_kwargs))
            spatial_loss_kwargs['loss_scales'] = spatial_decoder_kwargs.get('loss_scales', {'proj_depth':1.0, 'proj_hue':1.0, 'proj_normals':1.0})
            for layer in target_layers:
                d_attrs[layer] = outputs[layer]
            
            spatial_attrs_loss += rendering_particles_loss(
                logits=[d_attrs],
                labels={k:inp[:,:inp_sequence_len] for k,inp in inputs.items()},                
                particles_im_indices=spatial_inds,
                **spatial_loss_kwargs
            )

            # also try to learn projections of xyz onto hw
            if attrs_here[tier].get('pred_particles', None) is not None:
                spatial_loss_kwargs_hw = copy.deepcopy(spatial_loss_kwargs)
                spatial_loss_kwargs_hw['loss_scales'] = {'proj_xy': spatial_decoder_kwargs.get('proj_xy_loss', 1.0),
                                                         'proj_depth': spatial_loss_kwargs['loss_scales'].get('proj_depth', 1.0)}
                spatial_xy_loss = rendering_particles_loss(
                    logits=[attrs_here[tier]],
                    labels={k:inp[:,:inp_sequence_len] for k,inp in inputs.items()},                                    
                    particles_im_indices=None,
                    **spatial_loss_kwargs_hw)
                spatial_xy_loss = tf.Print(spatial_xy_loss, [spatial_xy_loss], message='spatial_xy_loss')
                spatial_attrs_loss += spatial_xy_loss
            # update losses from decoded attributes
            outputs.update({'decoded_attrs': decoded_attrs})
            outputs.update({'spatial_loss': {'spatial_loss':spatial_attrs_loss}})

        # decode predictions of future attributes
        if model_kwargs.get('compute_is_moving_loss', False) and train:
            which_nodes = model_kwargs.get('is_moving_nodes', 'spatial')
            nodes_here, segment_ids_here = _get_nodes_and_segments(outputs, which_nodes, take_every)
            tier = model_kwargs.get('is_moving_tier', 0)
            images = tf.cast(inputs['images'][:,:inp_sequence_len], tf.float32) / 255.
            future_loss = loss_functions.is_moving_loss(
                nodes=nodes_here[tier], segment_ids=segment_ids_here[tier], images=images, **future_decoder_kwargs)
            outputs.update({'future_loss': {'future_loss': future_loss}})            
        
        if model_kwargs.get('future_nodes', None) is not None and inp_sequence_len > 1 and not model_kwargs.get('compute_is_moving_loss', False):
            im_size = inputs[model_kwargs.get('images_key', 'images')].shape.as_list()[-3:-1]
            if train and (future_decoder_kwargs.get('num_motion_points', None) is not None):
                static_inds = sample_delta_image_inds(inputs['images'][:,:inp_sequence_len], num_points=future_decoder_kwargs['num_static_points'], static=True)
                motion_inds = sample_delta_image_inds(inputs['images'][:,:inp_sequence_len], num_points=future_decoder_kwargs['num_motion_points'])
                future_inds = tf.concat([static_inds, motion_inds], axis=2)
                motion_labels = tf.concat([tf.zeros(static_inds.shape.as_list()[:-1], tf.float32),
                                           tf.ones(motion_inds.shape.as_list()[:-1], tf.float32)], axis=-1)
            else:
                num_points = future_decoder_kwargs.get('num_decoder_points', 4096)
                future_inds = sample_image_inds(out_shape=[batch_size, len(output_times[::take_every])-1, num_points],
                                             im_size=im_size,
                                             train=train,
                                             **future_decoder_kwargs) # [B,T-1,P,2]
                motion_labels = tf.zeros(future_inds.shape.as_list()[:-1], tf.float32)
            
            which_nodes = model_kwargs.get('future_nodes', 'spatial')
            nodes_here, segment_ids_here = _get_nodes_and_segments(outputs, which_nodes, take_every)

            future_tiers = future_decoder_kwargs.get('tiers', [0])
            future_attrs_all = []
            future_loss = tf.cast(0.0, tf.float32)
            for tier in future_tiers:
                motion_logits = motion_classifier(
                    future_inds,
                    nodes_here[tier],
                    segment_ids_here[tier],
                    im_size=im_size,
                    **future_decoder_kwargs)
                future_loss += loss_functions.cross_entropy(motion_logits, motion_labels)
                
                if train and future_decoder_kwargs.get('depth_from_motion', False):
                    future_loss += depth_order_classifier(
                        motion_inds,
                        nodes_here[tier],
                        segment_ids_here[tier],
                        im_size=im_size,
                        **future_decoder_kwargs)
                
                future_attrs_all.append(motion_logits)

            outputs['future_attrs'] = future_attrs_all
            outputs.update({'future_loss': {'future_loss': future_loss}})
            
        # distribution loss: tries to estimate how much each node's attributes deviate from constant function
        if model_kwargs.get('spatial_distribution_kwargs', None) is not None:
            distribution_kwargs = model_kwargs.get('spatial_distribution_kwargs', {})
            which_nodes = model_kwargs.get('distribution_nodes', 'spatial')
            # attrs_here, segment_ids_here = _get_nodes_and_segments(outputs, which_nodes, take_every)
            if isinstance(encoder_outputs, list):
                attrs_here = encoder_outputs
                segment_ids_here = segment_ids
            else:
                attrs_here = encoder_outputs[which_nodes]
                if which_nodes == 'spatial':
                    segment_ids_here = segment_ids
                elif which_nodes == 'summary':
                    segment_ids_here = summary_segment_ids
            segment_ids_here = [tf.reshape(
                segs, [batch_size,len(output_times)]+segs.shape.as_list()[-2:])
                                for segs in segment_ids_here]
            
            distribution_tiers = distribution_kwargs.get('tiers', [0])
            distribution_labels = [dilate_tensor(inputs[key][:,:inp_sequence_len], take_every, axis=1) for key in distribution_kwargs.get('labels_keys', ['images'])]
            if 'valid' in inputs.keys():
                valid_images = dilate_tensor(inputs['valid'][:,:inp_sequence_len], take_every, axis=1)
            else:
                valid_images = None
            # for some reason tensorflow loses track of these shapes
            # segment_ids_here = [tf.reshape(
            #     segs, [batch_size,len(output_times)]+segs.shape.as_list()[-2:])
            #                     for segs in segment_ids_here]            

            if distribution_kwargs.get('hw_labels', False):
                im_shape = distribution_labels[0].shape.as_list()
                ones = tf.ones(im_shape[:-1], dtype=tf.float32)
                H,W = im_shape[-3:-1]
                him = tf.reshape((tf.range(H, dtype=tf.float32) / (H - 1.)) - 0.5, [1]*len(im_shape[:-3]) + [H,1]) * ones
                wim = tf.reshape((tf.range(W, dtype=tf.float32) / (W - 1.)) - 0.5, [1]*len(im_shape[:-3]) + [1,W]) * ones
                hw_labels = tf.stack([him, wim], axis=-1) # [...,H,W,2]
                distribution_labels.append(hw_labels)
            
            distribution_loss = tf.cast(0.0, tf.float32)
            for tier in distribution_tiers:
                valid_vectors = attrs_here[tier]['valid']
                distribution_loss += loss_functions.attr_variance_loss(
                    images_list=distribution_labels,
                    segments=segment_ids_here[tier],
                    attrs=attrs_here[tier]['vector'],
                    valid_attrs=valid_vectors,
                    valid_images=valid_images,
                    **distribution_kwargs)

            # distribution_loss = tf.Print(distribution_loss, [distribution_loss], message='distrib_loss_total')
            outputs.update({'distribution_loss': {'distribution_loss': distribution_loss}})
            

    if relative_attr_loss_kwargs is not None:
        ral_kwargs = copy.deepcopy(relative_attr_loss_kwargs)
        num_points = ral_kwargs.get('sample_points', 300)
        im_size = inputs[model_kwargs.get('images_key', 'images')].shape.as_list()[-3:-1]
        relative_tiers = ral_kwargs.get('tiers', [0])

        # get all the node pairs, includes invalid ones
        which_nodes = model_kwargs.get('relative_attr_nodes', 'spatial')
        hw_dims = ral_kwargs.get('hw_dims', [-4,-2])
        sample_nodes = [outputs[which_nodes + '_nodes'][tier] for tier in relative_tiers] # [B,T,N,D]
        if num_points is not None:
            sample_nodes = [s_nodes[:,:,:num_points] for s_nodes in sample_nodes] # take only first P
        sample_inds = [hw_attrs_to_image_inds(sample_nodes[tier][...,hw_dims[0]:hw_dims[1]], im_size)
                       for tier in relative_tiers] # list of [B,T,P,2] <tf.int32>
        # valid_inds = [tf.cast(s_nodes[...,-1:] > 0.5, tf.int32) for s_nodes in sample_nodes]

        # which attrs to decode
        attr_list = ral_kwargs.get('attr_list', ['depths'])
        attr_gt_preprocs_list = ral_kwargs.get('attr_gt_preprocs_list', [lambda d: tf.cast(-d, tf.float32) / 1000.0])
        attr_metrics_list = ral_kwargs.get('attr_metrics_list', [tf.subtract])
        gt_attr_metrics_list = ral_kwargs.get('gt_attr_metrics_list', attr_metrics_list)        
        attr_loss_kwargs_list = ral_kwargs.get('attr_loss_kwargs_list', [{'eps':0.1, 'invert':False}])
        relative_attr_loss = tf.cast(0.0, tf.float32)
        for i,im_inds in enumerate(sample_inds): # iter over tiers
            for t in range(len(output_times)): # time

                # special unsupervised depth-normals consistency loss
                if ral_kwargs.get('depth_normals_consistency_kwargs', None) is not None:
                    dn_loss = loss_functions.depth_normals_consistency_loss(
                        sample_nodes[i][:,t], **ral_kwargs['depth_normals_consistency_kwargs'])
                    dn_loss = tf.Print(dn_loss, [dn_loss], message='dn_cons_loss')
                    relative_attr_loss += dn_loss                

                # inds for supervised diffs
                nearest_k_inds = find_nearest_k_node_inds(
                    sample_nodes[i][:,t], nn_dims=hw_dims, **ral_kwargs) # [B,P,K]                
                b_inds = tf.reshape(tf.range(batch_size, dtype=tf.int32), [-1,1,1]) * tf.ones_like(nearest_k_inds)
                pred_inds = tf.stack([b_inds, nearest_k_inds], axis=-1)
                
                # get the pred diffs
                pred_attr_diffs_list, valid_diffs = attr_diffs_from_neighbor_inds(
                    sample_nodes[i][:,t], nearest_k_inds, **ral_kwargs)

                # find the valid points in the iamge
                gt_valid = inputs.get('valid', tf.ones(shape=(inputs[attr_list[0]].shape.as_list()[:-1] + [1]), dtype=tf.bool))
                gt_valid = gt_valid[:,t//take_every] if len(gt_valid.shape.as_list()) > 4 else gt_valid
                gt_valid = get_image_values_from_indices(gt_valid[:,tf.newaxis], im_inds[:,t:t+1])[:,0]
                gt_valid_neighbors = tf.cast(tf.gather_nd(gt_valid, pred_inds), tf.float32) # [B,P,K,1]
                # now iterate over attrs
                for j,attr in enumerate(attr_list):
                    
                    # get the gt diffs
                    # gt_images = inputs[attr][:,t//time_dilation] if len(inputs[attr].shape.as_list()) > 4 else inputs[attr]
                    gt_images = inputs[attr][:,t//take_every] if len(inputs[attr].shape.as_list()) > 4 else inputs[attr]                    
                    gt_images = attr_gt_preprocs_list[j](gt_images)
                    print("gt images", attr, gt_images.shape.as_list())
                    gt_attrs = tf.cast(get_image_values_from_indices(gt_images[:,tf.newaxis], im_inds[:,t:t+1])[:,0], tf.float32)  # [B,P,Dattr]
                    gt_attr_neighbors = tf.cast(tf.gather_nd(
                        # gt_attrs, tf.stack([b_inds, nearest_k_inds], axis=-1)), tf.float32) # [B,P,K,Dattr]
                        gt_attrs, pred_inds), tf.float32) # [B,P,K,Dattr]                    
                    gt_attr_diffs = gt_attr_metrics_list[j](gt_attrs[:,:,tf.newaxis,:], gt_attr_neighbors) # [B,P,K,1]

                    # compute loss between gt and pred diffs
                    valid_diffs_attr = valid_diffs * gt_valid_neighbors if attr != 'images' else valid_diffs
                    attr_loss = loss_functions.attr_diffs_loss(
                        pred_attr_diffs_list[j], gt_attr_diffs, valid_diffs_attr, **attr_loss_kwargs_list[j])

                    # attr_loss = tf.Print(attr_loss, [attr_loss], message=("rel_%s_loss_%d" % (attr, t)))
                    # relative_attr_loss += attr_loss
                    relative_attr_loss += attr_loss / tf.cast(len(output_times), tf.float32)                   
                    
        outputs['relative_attr_loss'] = {'relative_attr_loss': relative_attr_loss}

    # compute loss on edges
    if edges_error_signal is not None:
        edge_tier = model_kwargs.get('edge_tier', 0)
        edge_logits = outputs['errors'][edge_tier] # [B,T,N,1+kNN,4]
        base_logits, edge_logits = tf.split(edge_logits, [1,-1], axis=3)
        print("base and edge_logits", base_logits, edge_logits)
        base_inds, _, base_valid = tf.split(base_logits, [2,1,1], axis=-1) # [B,T,N,1,2/1/1]
        edge_inds, edge_logits, edge_valid = tf.split(edge_logits, [2,1,1], axis=-1) # [B,T,N,kNN,2/1/1]

        # now compute supervised or self-supervised loss
        if edges_error_signal == 'objects':
            objects = dilate_tensor(inputs[edges_error_signal][:,:inp_sequence_len], take_every, axis=1)
            im_size = objects.shape.as_list()[-3:-1]
            base_inds = hw_attrs_to_image_inds(base_inds, im_size) # [B,T,N,1,2]
            edge_inds = hw_attrs_to_image_inds(edge_inds, im_size) # [B,T,N,kNN,2]
            edge_labels = loss_functions.build_pairwise_segment_labels(objects, base_inds, edge_inds) # [B,T,N,kNN]

            edge_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=edge_labels, logits=edge_logits[...,0]) # [B,T,N,kNN]
            edge_loss = edge_loss * base_valid[...,0] * edge_valid[...,0]
            edge_loss = tf.Print(edge_loss, [
                tf.reduce_sum(tf.cast((edge_valid * tf.nn.sigmoid(edge_logits)) > 0.5, tf.float32)),
                tf.reduce_sum(edge_labels * edge_valid[...,0]),
                tf.reduce_sum(tf.ones_like(edge_labels * edge_valid[...,0]))], message='edge_preds_gt_max')
            edge_loss = model_kwargs.get('edges_loss_scale', 100.0) * tf.reduce_mean(edge_loss)
            outputs['vae_loss'] = {'vae_loss': edge_loss}
            

    if imnet_decoder is not None:
        imnet_outputs = [G_tnn.node[imnet_layer]['outputs'][t] for t in output_times]
        outputs['times'] = {t: imnet_outputs[i] for i,t in enumerate(output_times)}
        imnet_outputs = tf.stack(imnet_outputs, axis=1)
        # imnet_outputs = tf.stack([G_tnn.node[imnet_layer]['outputs'][t] for t in output_times], axis=1)
        print("imnet outputs", imnet_outputs.shape.as_list())
        imnet_logits = imnet_decoder(imnet_outputs, **imnet_decoder_kwargs)
        print("imnet_logits", imnet_logits)
        outputs['times']['dec'] = imnet_logits

        if inputs.get('imagenet_labels', None) is not None:
            imnet_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=imnet_logits,
                    labels=inputs['imagenet_labels']
                ))
            outputs.update({
                'imagenet_logits': imnet_logits,
                'imagenet_loss': {'imagenet_loss': imnet_loss}
            })

    return outputs, params
    
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
