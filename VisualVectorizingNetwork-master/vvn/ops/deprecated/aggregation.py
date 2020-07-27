import tensorflow as tf
import numpy as np

lp = tf.load_op_library('../../ops/src/tf_labelprop.so')
lpfc = tf.load_op_library('../../ops/src/tf_labelprop_fc.so')
hung = tf.load_op_library('../../ops/src/hungarian.so')

def agg_features_from_segments(features, segment_ids, num_segments,
                               max_segments_per_example=256,
                               augmentation_kernel_list=[],
                               agg_diffs=False,
                               stop_gradient=False,
                               augmentation_inds=None,
                               normalize_coordinates=True,
                               normalize_areas=True,
                               return_channels_dict=False
):
    '''
    Aggregate features from segment ids, e.g. as computed by label prop
    for the C original feature channels in features, aggregate by averaging.

    Also compute the centroid (average (h,w) location) for each segment and optional 
    aggregate statistics.

    Then reformat summary features into a [B,N_max,C'] tensor where examples with 
    fewer than N_max segments are padded with zeros and indicated as fake.

    features: [B,H,W,C] <tf.float32>
    segment_ids: [B,HW] <tf.int32>
    num_segments: [B] <tf.int32>
    max_segments_per_example: int Nmax, determines shape of output tensor

    outputs:
    summary_features: [B,Nmax,C'] where C' = C + 2 (h,w coordinate means) + 1 (segment area) + kC (optional derived features)
    '''
    B,H,W,C = features.shape.as_list()
    channels_dict = OrderedDict() # to indicate what features go where
    new_channels = 0
    channels_dict['input'] = [0,C]
    sg_func = tf.stop_gradient if stop_gradient else tf.identity

    # feature augmentation
    if len(augmentation_kernel_list):
        aug_features, aug_new_channels = augment_features(features, augmentation_kernel_list, channel_inds=augmentation_inds)
        features = tf.concat([features] + [sg_func(aug) for aug in aug_features], axis=-1)
        new_channels += aug_new_channels
        channels_dict['aug'] = [C, C+aug_new_channels]
    else:
        aug_new_channels = 0

    # prepend batch, height, width indices
    ones = tf.ones([B,H,W,1], dtype=tf.float32)
    b_inds = tf.reshape(tf.range(B, dtype=tf.float32), [B,1,1,1]) * ones
    h_inds = tf.reshape(tf.range(H, dtype=tf.float32), [1,H,1,1]) * ones
    w_inds = tf.reshape(tf.range(W, dtype=tf.float32), [1,1,W,1]) * ones

    if normalize_coordinates:
        h_inds = (h_inds / ((H-1.0)/2.0)) - 1.0
        w_inds = (w_inds / ((W-1.0)/2.0)) - 1.0

    features = tf.concat([ones, b_inds, h_inds, w_inds, features], axis=-1)
    new_channels += 2 # only H,W channels will be summed over, not ones,B

    # reshape for aggregation
    features = tf.reshape(features, [B*H*W, C + new_channels + 2])

    # aggregate features
    if PRINT and (max_segments_per_example > 1):
        num_segments = tf.Print(num_segments, [num_segments, tf.constant(max_segments_per_example)], message="num_segments")
    Ntotal = tf.reduce_sum(num_segments)
    mean_feats = tf.math.unsorted_segment_sum(
        features,
        segment_ids=tf.reshape(segment_ids, [-1]),
        num_segments=Ntotal
    )
    segment_areas, mean_feats = tf.split(mean_feats, [1,-1], axis=-1)
    mean_feats = mean_feats / tf.maximum(1.0, segment_areas)
    b_inds, mean_feats = tf.split(mean_feats, [1,-1], axis=-1) # first 2 channels now centroid

    # optionally do more types of aggregation (skip ones and b_inds)
    if agg_diffs:
        ss_feats = tf.math.unsorted_segment_mean(
            tf.square(features[...,2:]), segment_ids=tf.reshape(segment_ids, [-1]), num_segments=Ntotal)
        var_feats = ss_feats - tf.square(mean_feats)

        agg_feats = sg_func(var_feats)
        mean_feats = tf.concat([mean_feats, agg_feats], axis=-1)
        new_channels += (C + aug_new_channels + 2)

    # get indicies for reshaping outputs into [B,Nmax,C]
    Nmax = max_segments_per_example

    b_inds = tf.cast(b_inds, tf.int32) # [Ntotal,1]
    s_inds = within_example_segment_ids(num_segments, Nmax)[:,tf.newaxis] # [Ntotal, 1]

    # append "metadata" of each segment
    valid_summaries = tf.ones([Ntotal,1], dtype=tf.float32)
    if normalize_areas:
        segment_areas = segment_areas / (1.0 * H * W)
    summary_feats = tf.concat([mean_feats, segment_areas, valid_summaries], axis=-1) # [Ntotal, C+4]
    new_channels += 2 # centroids are first channels, segment areas and valid flag are last

    # scatter into rectangular summary features
    summary_feats = tf.scatter_nd(
        tf.concat([b_inds,s_inds], axis=-1), # [Ntotal,2]
        summary_feats, # [Ntotal,C+3]
        shape=[B,Nmax,C+new_channels]
    )

    if return_channels_dict:
        return summary_feats, channels_dict
    else:
        return summary_feats

def agg_attrs_across_nodes(nodes, labels, num_segments, max_labels=32,
                           get_range_attrs=True, get_var_attrs=True,
                           rectangular_output=True, **kwargs):
    '''
    aggregation
    '''
    B,N,D = nodes.shape.as_list()
    Ntotal = tf.reduce_sum(num_segments)

    # aggregate mean attrs for valid nodes
    valid_inds = tf.where(nodes[...,-1] > 0.5)
    real_nodes = tf.gather_nd(nodes, valid_inds) # [?,D]
    sum_attrs = tf.math.unsorted_segment_sum(real_nodes, labels, num_segments=Ntotal)
    num_nodes_attr = sum_attrs[:,-1:]
    mean_attrs = sum_attrs / tf.maximum(num_nodes_attr, 1.0)

    # add new attributes via aggregation statistics
    new_channels = 2
    valid_attr = tf.ones_like(num_nodes_attr)
    attrs_list = [mean_attrs, num_nodes_attr, valid_attr]

    if get_range_attrs:
        min_attrs = tf.math.unsorted_segment_min(real_nodes, labels, num_segments=Ntotal)
        max_attrs = tf.math.unsorted_segment_max(real_nodes, labels, num_segments=Ntotal)
        range_attrs = max_attrs - min_attrs
        attrs_list.insert(-2, range_attrs)
        new_channels += D
    if get_var_attrs:
        ss_attrs = tf.math.unsorted_segment_mean(tf.square(real_nodes), labels, num_segments=Ntotal)
        var_attrs = ss_attrs - tf.square(mean_attrs)
        attrs_list.insert(-2, var_attrs)
        new_channels += D

    # now concat all aggregated attrs
    agg_attrs = tf.concat(attrs_list, axis=-1) # [Ntotal, D+D']
    b_inds, n_inds = inds_from_num_segments(num_segments, max_labels)
    inds = tf.stack([b_inds, n_inds], axis=-1) # [Ntotal,2]
    if rectangular_output:
        # reshape into a rectangular tensor
        agg_nodes = tf.scatter_nd(inds, agg_attrs, shape=[B,max_labels,D+new_channels])
        valid_attr = tf.cast(tf.logical_and(agg_nodes[...,-1:] < 1.1, agg_nodes[...,-1:] > 0.9), tf.float32)
        agg_nodes = tf.concat([agg_nodes[...,:-1], valid_attr], axis=-1)
        return agg_nodes
    else:
        return agg_attrs, inds

def spatial_node_inference_from_spatial_features(features, labels=None, num_segments=None, focal_lengths=None,
                                                 labelprop_k=2, num_steps=10, defensive=False, Nmax=128,
                                                 labelprop_dims_list=None,
                                                 metric=euclidean_dist2,
                                                 metric_kwargs={'thresh':None},
                                                 symmetric=False,
                                                 synchronous=False,
                                                 mode='index',
                                                 node_dim=20,
                                                 mlp_kwargs={'hidden_dims':[], 'activation': tf.nn.swish},
                                                 augmentation_kernel_list=[], agg_diffs=False, stop_gradient=False,
                                                 modify_areas=True,
                                                 init_focal_length=2.0,
                                                 **kwargs
):
    '''
    Input spatial features [B,H,W,C]
    Output summary nodes of shape [B, Nmax, out_dim] via labelprop
    '''
    B,H,W,C = features.shape.as_list()
    size = tf.constant([H,W], tf.int32)
    if labelprop_dims_list is None:
        labelprop_dims_list = [[0,C]] # use all features

    if labels is None:
        if Nmax > 1: # use labelprop to get summary nodes
            lp_features = tf.concat([features[...,ld[0]:ld[1]] for ld in labelprop_dims_list], axis=-1)
            adjacency = compute_adjacency_from_features(
                lp_features, k=labelprop_k, metric=metric, metric_kwargs=metric_kwargs, symmetric=symmetric)
            labels, num_segments = compute_segments_by_label_prop(
                adjacency, size, num_steps=num_steps, defensive=defensive, synchronous=synchronous, mode=mode)
        elif Nmax == 1: # global aggregation (e.g. for a scene node)
            labels = tf.reshape(tf.range(B, dtype=tf.int32), [B,1]) * tf.ones([B,H*W], dtype=tf.int32)
            num_segments = tf.ones([B], dtype=tf.int32)

    # agg features
    summary_features = agg_features_from_segments(
        features, labels, num_segments, max_segments_per_example=Nmax,
        augmentation_kernel_list=augmentation_kernel_list, agg_diffs=agg_diffs,
        stop_gradient=stop_gradient, normalize_coordinates=True, normalize_areas=True
    ) # [B,Nmax,C']

    # use mlp to get node attributes per summary feature vector
    hw_centroids, summary_features, valid_nodes = tf.split(summary_features, [2,-1,1], axis=-1)
    area_attr = summary_features[...,-1:] # the second to last channel during aggregation was (normalized) area
    node_attrs = shared_spatial_mlp(summary_features, out_depth=(node_dim-5), # first 2, last 4 attrs are xy, hwav
                                         scope="features_to_attrs_mlp",
                                         **mlp_kwargs
    )
    z_attr, area_modifier, node_attrs = tf.split(node_attrs, [1,1,-1], axis=-1)

    # get xy through inverse camera projection on initial estimate of z
    if focal_lengths is None:
        f = tf.get_variable(name="focal_length", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(value=init_focal_length))
        f = tf.nn.relu(f)
    else:
        f = focal_lengths
    f = tf.reshape(f, [-1,1,1])
    # f = tf.Print(f, [f, f.shape], message='focal_length')
    yx_attrs = hw_centroids * tf.div(-z_attr, tf.maximum(f, 0.01)) # inverse projection; z is negative
    xy_attrs = tf.stack([yx_attrs[...,1], -yx_attrs[...,0]], axis=-1) # flip y coordinate b/c of matrix indexing top to bottom

    # true area will be a function of depth, pixel area, and normal vectors/curvature
    if modify_areas:
        area_attr = area_attr * tf.maximum(-z_attr, 1.0) * tf.nn.relu(area_modifier)

    # attr to say which nodes are real (i.e. were scattered from a single segment)
    valid_attr = tf.cast(tf.logical_and(tf.less(valid_nodes, 1.1), tf.greater(valid_nodes, 0.9)), tf.float32)

    # output has node_dim, order [xy, z, node_dim-7, hw, a, v]
    spatial_nodes = tf.concat([xy_attrs, z_attr, node_attrs, hw_centroids, area_attr, valid_attr], axis=-1)

    return spatial_nodes, summary_features, labels, num_segments

def spatial_nodes_to_features(nodes, valid_nodes, segment_ids, out_channels, ksize=1, **kwargs):
    B,N,D = nodes.shape.as_list()
    _,H,W = segment_ids.shape.as_list()
    C = out_channels

    # preproc segment_ids
    segment_ids -= tf.reduce_min(segment_ids, axis=[1,2], keepdims=True) # start at 0 per example
    segment_ids = tf.where(segment_ids < N,
                           segment_ids,
                           tf.ones_like(segment_ids)*(N-1)) # max of N-1

    # gather node values
    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1]) * tf.ones_like(segment_ids) # [B,H,W]
    nodes_to_features = tf.gather_nd(nodes * valid_nodes, tf.stack([b_inds, segment_ids], axis=-1)) # [B,H,W,D]

    # conv to output shape
    kernel_initializer = tf.variance_scaling_initializer(seed=0, scale=0.001)
    nodes_to_features = _conv(nodes_to_features, [ksize,ksize], out_depth=C, scope="nodes_to_features_conv", kernel_initializer=kernel_initializer)

    return nodes_to_features

def summary_node_inference_from_spatial_nodes(
        spatial_nodes, edges=None,
        num_summary_nodes=32,
        labelprop_kwargs={'metric_kwargs': {'thresh':'mean', 'thresh_scale':0.25}, 'num_steps':10, 'synchronous':False},
        aggregation_kwargs={}):

    if edges is None:
        labels, num_segments = labels_from_nodes(spatial_nodes, **labelprop_kwargs)
    elif edges is not None: # passed a boolean mat of edgse
        labels, num_segments = labels_from_nodes_and_edges(spatial_nodes, edges, **labelprop_kwargs)

    if False:
        num_segments = tf.Print(num_segments, [num_segments], message='summary_num_segments')
    summary_nodes = agg_attrs_across_nodes(spatial_nodes, labels, num_segments, max_labels=num_summary_nodes, **aggregation_kwargs)
    return summary_nodes, labels, num_segments


def feature_map_from_segments_and_nodes(segment_ids, node_features):
    '''
    segment_ids: [B,T,H,W] <tf.int32> labels unpreprocessed (so they range from [0, Ntotal)
    nodes: [B,T,N,C] <tf.float32> nodes to be indexed into with feature vectors in R^C

    returns: 
    feature_map [B,T,H,W,C] <tf.float32> given by indexing (i.e. the "paint-by-numbers" spatial decoder)
    '''

    B,T,H,W = segment_ids.shape.as_list()
    _,_,N,C = node_features.shape.as_list()

    # preproc segments
    segment_ids = segment_ids - tf.reduce_min(segment_ids, axis=[2,3], keepdims=True)
    segment_ids = tf.where(segment_ids < N,
                           segment_ids,
                           (N-1)*tf.ones_like(segment_ids))

    # build inds
    ones = tf.ones([B,T,H,W], dtype=tf.int32)
    b_inds = tf.reshape(tf.range(B, dtype=tf.int32), [B,1,1,1]) * ones
    t_inds = tf.reshape(tf.range(T, dtype=tf.int32), [1,T,1,1]) * ones
    gather_inds = tf.stack([b_inds, t_inds, segment_ids], axis=-1) # [B,T,H,W,3]
    feature_map = tf.gather_nd(node_features, gather_inds) # [B,T,H,W,C]

    return feature_map

def nodes_from_segments_and_feature_map(segment_ids, feature_map, num_segments=None, max_segments=128, agg_type='mean'):
    '''
    segment_ids: [B,[T],H,W] <tf.int32> labels assumed to be in order
    feature_map: [B,[T],H,W,C] <tf.float32> features to be aggregated
    '''
    seg_shape = segment_ids.shape.as_list()
    if len(seg_shape) == 4:
        B,T,H,W = seg_shape
        segment_ids = tf.reshape(segment_ids, [B*T,H,W])
    elif len(seg_shape) == 3:
        B,H,W = seg_shape
        T = 1
    else:
        raise ValueError("seg shape is %s but must be len 3 or 4" % seg_shape)

    assert seg_shape == feature_map.shape.as_list()[:-1], "feature map shape per channel must match segment_ids shape"

    C = feature_map.shape.as_list()[-1]
    N = max_segments
    if num_segments is None:
        num_segments = tf.reduce_max(segment_ids, axis=[1,2], keepdims=False) - tf.reduce_min(segment_ids, axis=[1,2]) + tf.constant(1, tf.int32)
    else:
        assert num_segments.shape.as_list() == [B*T]
    Ntotal = tf.reduce_sum(num_segments)

    if agg_type == 'mean':
        agg_func = tf.unsorted_segment_mean
    elif agg_type == 'sum':
        agg_func = tf.unsorted_segment_sum
    elif agg_type == 'max':
        agg_func = tf.unsorted_segment_max
    elif agg_type == 'min':
        agg_func = tf.unsorted_segment_min
    else:
        raise NotImplementedError()

    # aggregate
    nodes = agg_func(
        tf.reshape(feature_map, [-1,C]),
        segment_ids=tf.reshape(segment_ids, [-1]),
        num_segments=Ntotal) # [Ntotal, C]
    # scatter to rectangular
    scatter_inds = tf.stack(inds_from_num_segments(num_segments, N), axis=-1)
    nodes = tf.scatter_nd(
        scatter_inds, nodes, shape=[B*T,N,C])

    if len(seg_shape) == 4:
        nodes = tf.reshape(nodes, [B,T,N,C])

    return nodes
