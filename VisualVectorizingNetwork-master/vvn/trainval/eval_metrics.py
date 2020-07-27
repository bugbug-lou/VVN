from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sklearn.metrics
import skimage
from skimage.segmentation.boundaries import find_boundaries
from sklearn.cluster import KMeans

from vvn.data.data_utils import read_depths_image, object_id_hash
from vvn.models.rendering import hw_attrs_to_image_inds

class ObjectMetrics(object):
    '''
    A class for measuring object mask IoU and AP
    '''
    def __init__(self,
                 gt_objects,
                 pred_objects=None,
                 background_value=0,
                 min_gt_size=1,
                 stride=1,
                 val=256):
        self.gt_objects = gt_objects
        self.gt_shape = self.gt_objects.shape
        self.B, self.T, self.H, self.W = self.gt_shape[:4]
        # if gt objects not already hashed to unique values, do that
        if len(self.gt_shape) == 5 and self.gt_shape[-1] > 1:
            self.gt_objects = self._object_id_hash(self.gt_objects)
            self.gt_shape = self.gt_objects.shape
        # otherwise just take single channel
        elif len(self.gt_shape) == 5 and self.gt_shape[-1] == 1:
            self.gt_objects = self.gt_objects[...,0]
            self.gt_shape = self.gt_shape[:-1]
        else:
            assert len(self.gt_shape) == 4, self.gt_shape

        self.pred_objects = pred_objects

        # hyperparameters
        self.bg_val = background_value
        self.min_gt_size = min_gt_size
        self.stride = stride

        # preprocessing of gt masks and init metrics
        # the unique hashed gt object ids
        self.set_gt_ids()
        self.IoUs = None
        self.pred_object_ids = None

    def __setattr__(self, item, value):
        if item == 'pred_objects':
            if value is not None:
                shape = value.shape
                assert shape[:2] == self.gt_shape[:2], "pred objects must have the same shape[:2] as gt objects, %s, %s" % (value.shape, self.gt_shape)
                # resize
                if shape[-2:] != self.gt_shape[-2:]:
                    value = np.int32(skimage.transform.resize(value.astype(float), self.gt_shape, order=0))
                # obj ids start at 0 per example
                offsets = value.min(axis=(-2,-1), keepdims=True)
                value = value - offsets
            dict.__setattr__(self, item, value)
        else:
            dict.__setattr__(self, item, value)

    def set_gt_ids(self):
        '''
        set len(B) list of len(T) lists of unique ids per example, time
        '''
        self.gt_ids = []
        for ex in range(self.B):
            self.gt_ids.append(
                [np.unique(self.gt_objects[ex,t]) for t in range(self.T)])

    def get_gt_mask(self, ex, t, obj_id):
        mask = (self.gt_objects[ex,t] == obj_id)
        return mask

    @staticmethod
    def mask_IoU(pred_mask, gt_mask, min_gt_size=1):
        assert pred_mask.shape == gt_mask.shape, (pred_mask.shape, gt_mask.shape)
        assert pred_mask.dtype == gt_mask.dtype == np.bool, (pred_mask.dtype, gt_mask.dtype)
        num_gt_px = gt_mask.sum()
        num_pred_px = pred_mask.sum()
        if num_gt_px < min_gt_size:
            return np.nan

        overlap = (pred_mask & gt_mask).sum().astype(float)
        IoU = overlap / (num_gt_px + num_pred_px - overlap)
        return IoU

    @staticmethod
    def compute_aris(pred_segments, gt_segments):
        B,T,H,W = pred_segments.shape
        _,Tim,Him,Wim = gt_segments.shape
        strides = [Him // H, Wim // W]
        gt_segments = gt_segments[:,:,::strides[0],::strides[1]]
        assert pred_segments.shape == gt_segments.shape, (pred_segments.shape, gt_segments.shape)

        aris = np.zeros([B,T], dtype=np.float32)
        for b in range(B):
            for t in range(T):
                gt = gt_segments[b,t].reshape([-1])
                pred = pred_segments[b,t].reshape([-1])
                ari = sklearn.metrics.adjusted_rand_score(gt, pred)
                aris[b,t] = ari

        return aris

    def compute_best_IoUs(self, examples=None, times=None, **kwargs):
        assert self.pred_objects is not None, "Need predictions set to self.pred_objects to compute IoUs"
        examples = examples or range(self.B)
        times = times or range(self.T)
        best_IoUs = [[ [] for t in times] for ex in examples]
        best_pred_objs = [[ [] for t in times] for ex in examples]
        for i,ex in enumerate(examples):
            for j,t in enumerate(times):
                ids_here = [
                    oid for oid in self.gt_ids[ex][t] if oid != self.bg_val]
                best_IoUs_here = []
                best_pred_objs_here = []
                for oid in ids_here:
                    gt_mask = self.get_gt_mask(ex, t, oid)
                    preds = self.pred_objects[ex,t]
                    pred_masks = [
                        preds == k for k in range(preds.max() + 1)]
                    IoUs_here = [
                        self.mask_IoU(pred_mask, gt_mask, self.min_gt_size)
                        for pred_mask in pred_masks]
                    best_IoUs_here.append(max(IoUs_here))
                    best_pred_objs_here.append(
                        np.argmax(np.array(IoUs_here)))

                best_IoUs[i][j] = best_IoUs_here
                best_pred_objs[i][j] = best_pred_objs_here
                self.IoUs = best_IoUs
                self.pred_object_ids = best_pred_objs

        return best_IoUs, best_pred_objs

    def compute_recalls(self, examples=None, times=None, thresh=0.5, **kwargs):
        '''
        Proportion of ground truth object segments with a pred best match IoU > thresh
        '''
        examples = examples or range(self.B)
        times = times or range(self.T)
        if self.IoUs is None:
            IoUs, _ = self.compute_best_IoUs(examples, times)
        else:
            IoUs = [[self.IoUs[ex][t] for t in times] for ex in examples]
        recalls = np.zeros(shape=[len(examples), len(times)], dtype=np.float32)

        for i,ex in enumerate(examples):
            for j,t in enumerate(times):
                true_pos = np.array(IoUs[i][j]) >= thresh
                recall = (true_pos.sum().astype(float) / len(true_pos)) if len(true_pos) else np.nan
                recalls[i][j] = recall

        return recalls[...,np.newaxis]

    def compute_boundary_metrics(self, examples=None, times=None, stride=1, connectivity=2, mode='thick', compute_BDEs=True, normalize=True, **kwargs):
        '''
        For matched pred and gt masks, compute F measure on their boundary pixels.
        F measure is defined as 2*(precision * recall) / (precision + recall)
        '''
        examples = examples or range(self.B)
        times = times or range(self.T)
        s = stride

        boundary_F1s = [[ [] for t in times] for ex in examples]
        boundary_DEs = [[ [] for t in times] for ex in examples]
        for i,ex in enumerate(examples):
            for j,t in enumerate(times):
                gt_masks, pred_masks = self.get_gt_and_pred_masks(ex,t)
                num_objs = len(gt_masks)

                F1s_here = []
                BDEs_here = []
                for obj in range(num_objs):
                    gt_boundary = find_boundaries(gt_masks[obj][::s,::s], connectivity=connectivity, mode=mode)
                    pred_boundary = find_boundaries(pred_masks[obj][::s,::s], connectivity=connectivity, mode=mode)

                    # precision and recall and F1
                    true_pos = (gt_boundary & pred_boundary).sum().astype(float)
                    false_pos = (~gt_boundary & pred_boundary).sum().astype(float)
                    false_neg = (gt_boundary & (~pred_boundary)).sum().astype(float)
                    precision = true_pos / (true_pos + false_pos) if (true_pos > 0.0) else 1.0 - (false_pos > 0.0).astype(float)
                    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg > 0.0) else 1.0
                    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall > 0.0) else 0.0
                    F1s_here.append(F1)

                    # boundary displacement error
                    if compute_BDEs:
                        pred_px = np.where(pred_boundary)
                        gt_px = np.where(gt_boundary)

                        dists = []
                        for p in zip(pred_px[0], pred_px[1]):
                            dists_p = [np.sqrt((p[0] - g[0])**2 + (p[1] - g[1])**2)
                                       for g in zip(gt_px[0], gt_px[1])]
                            if len(dists_p):
                                dists.append(min(dists_p))
                            else:
                                dists.append(0.0)
                        dists = np.array(dists) / (1.0 if not normalize else pred_boundary.shape[0])
                        BDEs_here.append((np.nanmean(dists), np.std(dists)))

                #update
                boundary_F1s[i][j].extend(F1s_here)
                boundary_DEs[i][j].extend(BDEs_here)

        self.BFMs = boundary_F1s
        self.BDEs = boundary_DEs
        return boundary_F1s, boundary_DEs

    def cluster_unlabeled_pixels(self, features, n_clusters=12, stride=1):
        '''
        Compute "background" segments by clustering features that aren't already in a segment
        '''
        fg = np.int32(self.gt_objects > self.bg_val)
        bg = 1-fg
        # reset so background is gone
        if features is None:
            self.gt_objects = self.gt_objects * fg + (self.bg_val * bg)
            self.set_gt_ids()
            return

        # else cluster
        assert features.shape[:-1] == self.gt_objects.shape
        features = features[:,:,::stride,::stride]
        bg_labels = self.bg_val * np.ones_like(self.gt_objects[:,:,::stride,::stride])
        for ex in range(features.shape[0]):
            for t in range(features.shape[1]):
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                labels = kmeans.fit(features[ex,t].reshape([-1,features.shape[-1]])).labels_ + 1
                bg_labels[ex,t] = self.bg_val - labels.reshape(features.shape[2:4])

        # resize
        bg_labels = np.int32(skimage.transform.resize(bg_labels.astype(float), self.gt_shape, order=0))

        # update
        self.gt_objects = self.gt_objects*fg + bg_labels*bg
        self.set_gt_ids()
        return

    def get_gt_and_pred_masks(self, ex, t):
        assert type(ex) == type(t) == int
        if self.pred_object_ids is None:
            _, objs = self.compute_best_IoUs([ex],[t])
            objs = objs[0][0]
        else:
            objs = self.pred_object_ids[ex][t]

        obj_ids = [oid for oid in self.gt_ids[ex][t] if oid != self.bg_val]
        gt_masks = []
        pred_masks = []
        for i, obj in enumerate(objs):
            gt_mask = self.get_gt_mask(ex,t,obj_ids[i])
            gt_masks.append(gt_mask)
            pred_mask = self.pred_objects[ex,t] == obj
            pred_masks.append(pred_mask)

        return gt_masks, pred_masks

    @staticmethod
    def mean_across_objects(objs_per_time_per_ex):
        B = len(objs_per_time_per_ex)
        T = len(objs_per_time_per_ex[0])
        D = 1
        for ex in range(B):
            for t in range(T):
                val = objs_per_time_per_ex[ex][t]
                if len(val):
                    try:
                        D = np.maximum(len(val[0]), D)
                    except TypeError:
                        D = 1


        results = np.zeros(shape=[B,T,D], dtype=np.float32)
        for b in range(B):
            for t in range(T):
                for d in range(D):
                    results[b,t,d] = np.nanmean(np.array(
                        [(val[d] if D > 1 else val) for val in objs_per_time_per_ex[b][t]]))

        return results

    def compute_all_metrics(self, examples=None, times=None, **kwargs):
        results = {}
        examples = examples or range(self.B)
        times = times or range(self.T)

        # mean mask IoU
        IoUs, pred_obj_ids = self.compute_best_IoUs(examples, times)
        gt_obj_ids = [[self.gt_ids[ex][t] for t in times] for ex in examples]
        results['mIoU'] = self.mean_across_objects(IoUs)
        results['pred_object_ids'] = pred_obj_ids
        results['gt_object_ids'] = self._invert_object_id_hash(gt_obj_ids)

        # mask overlapp recall at kwargs.get('thresh', 0.5)
        recalls = self.compute_recalls(examples, times, **kwargs)
        results['recall'] = recalls

        bfms, bdes = self.compute_boundary_metrics(
            examples, times, **kwargs)
        results['boundary_f_measure'] = self.mean_across_objects(bfms)
        if kwargs.get('compute_BDEs', False):
            results['boundary_displacement_error'] = self.mean_across_objects(bdes)

        # ari
        if kwargs.get('compute_ARIs', False):
            gt_segs = self.gt_objects[:,:,::self.stride,::self.stride]
            pred_segs = self.pred_objects[:,:,::self.stride,::self.stride]
            results['adjusted_rand_index'] = self.compute_aris(pred_segs, gt_segs)[...,np.newaxis]

        return results

    def _object_id_hash(self, objects, dtype_out=np.int32, val=256):
        C = objects.shape[-1]
        out = np.zeros(shape=objects.shape[:-1], dtype=dtype_out)
        for c in range(C):
            scale = np.power(val, C-1-c)
            out += scale * objects[...,c]
        return out

    def _invert_object_id_hash(self, object_ids, dtype_out=np.uint8, val=256, out_channels=3):
        C = out_channels
        B = len(object_ids)
        T = len(object_ids[0])
        object_id_tuples = [[ [] for t in range(T)] for ex in range(B)]

        for b in range(B):
            for t in range(T):
                ids_list = [oid for oid in object_ids[b][t] if oid != self.bg_val]
                ids_tuples = []
                for oid in ids_list:
                    obj_tup = [
                       dtype_out((oid % val**(C-c)) // (val**(C-1-c))) for c in range(C)
                    ]
                    ids_tuples.append(obj_tup)
                object_id_tuples[b][t].extend(ids_tuples)

        return object_id_tuples


def get_pred_and_gt_segments(inputs, outputs, target, take_every=1, gt_key='objects', segments_key='summary_segment_ids',
                             bg_keys=[], depths_preproc=read_depths_image, **kwargs):
    '''
    target should be 'objects' for Gibson

    '''

    tier = kwargs.get('tier', 0)
    print("segments_key", segments_key)
    pred_segments = outputs[segments_key][tier][:,take_every-1::take_every]
    num_times = pred_segments.shape.as_list()[1]

    t_offset = kwargs.get('t_offset', 0)
    gt_segments = inputs[gt_key][:,t_offset:t_offset+num_times]
    gt_segments = object_id_hash(gt_segments, dtype_out=tf.int32, val=256)[...,0] # [B,T,Him,Wim]

    results = {'pred': pred_segments, 'gt': gt_segments}
    if kwargs.get('compute_BDEs', 0):
        results['bde_flag'] = tf.cast(1, tf.bool)
    if kwargs.get('compute_ARIs', 0):
        results['ari_flag'] = tf.cast(1, tf.bool)
    results['thresh'] = tf.cast(kwargs.get('recall_thresh', 0.5), tf.float32)
    results['agg_mean'] = tf.cast(kwargs.get('agg_mean', 0), tf.bool)

    # features for finding background segments
    bg_feats = []
    if 'normals' in bg_keys:
        normals = inputs['normals']
        bg_feats.append(normals)
    if 'xyz' in bg_keys:
        pmat = inputs['projection_matrix']
        z = depths_preproc(inputs['depths'])
        B,T,H,W,_ = z.shape.as_list()
        ones = tf.ones_like(z)
        wh = tf.stack([
            tf.tile(tf.linspace(-1.0, 1.0, W)[tf.newaxis,:], [H,1]),
            -tf.tile(tf.linspace(-1.0, 1.0, H)[:,tf.newaxis], [1,W])
        ], axis=-1) # [H,W,2]
        wh = wh[tf.newaxis,tf.newaxis] * ones # [B,T,H,W,2]
        f = tf.stack([pmat[:,:,0,0], pmat[:,:,1,1]], axis=-1)[:,:,tf.newaxis,tf.newaxis] * ones # [B,T,H,W,2]
        xy = (wh * z) / f
        xyz = tf.concat([xy,z], axis=-1)
        bg_feats.append(xyz)

    if len(bg_feats):
        bg_feats = tf.concat([tf.cast(bg, tf.float32) for bg in bg_feats], axis=-1)
    else:
        bg_feats = tf.cast(0, tf.bool)
    results['background_features'] = bg_feats

    # other kwargs for metrics func
    results['background_n_clusters'] = tf.cast(kwargs.get('background_n_clusters', 5), tf.uint8)
    return results

def get_segments_and_edges(inputs, outputs, target, take_every=1, **kwargs):

    results = get_pred_and_gt_segments(inputs, outputs, target, **kwargs)

    edge_tier = kwargs.get('edge_tier', 0)
    edge_logits = outputs['errors'][edge_tier][:,take_every-1::take_every] # [B,T,N,1+kNN,4]
    num_times = edge_logits.shape.as_list()[1]
    base_logits, edge_logits = tf.split(edge_logits, [1,-1], axis=3)
    base_inds, _, base_valid = tf.split(base_logits, [2,1,1], axis=-1) # [B,T,N,1,2/1/1]
    edge_inds, edge_logits, edge_valid = tf.split(edge_logits, [2,1,1], axis=-1) # [B,T,N,kNN,2/1/1]
    objects = inputs['objects'][:,:num_times]
    im_size = objects.shape.as_list()[-3:-1]
    base_inds = hw_attrs_to_image_inds(base_inds, im_size) # [B,T,N,1,2]
    edge_inds = hw_attrs_to_image_inds(edge_inds, im_size) # [B,T,N,kNN,2]

    edge_labels = tf.cast(build_pairwise_segment_labels(objects, base_inds, edge_inds), tf.bool) # [B,T,N,kNN]
    edge_preds = tf.nn.sigmoid(edge_logits[...,0]) > 0.5

    correct_edges = tf.cast(tf.equal(edge_labels, edge_preds), tf.float32) * edge_valid[...,0] * base_valid[...,0]
    edge_accuracy = tf.reduce_sum(correct_edges) / tf.maximum(1.0, tf.reduce_sum(edge_valid[...,0] * base_valid[...,0]))
    results['edge_accuracy'] = edge_accuracy
    return results

def compute_summary_image_mses(inputs, outputs, target,
                               label_to_attr_keys={'images': 'pred_colors'},
                               label_preprocs={'images': lambda rgb: tf.image.rgb_to_hsv(rgb / 255.)},
                               im_size=[64,64],
                               max_depth=None,
                               **kwargs):

    '''
    Get rgb, depth, normals, or other MSEs for which there are pixel-level labels

    inputs: dict of tensor inputs from data provider
    outputs: dict of model outputs
    target: a list of keys into images to get the labels
    attr_keys: a dict of {label_name: attr_name} pairs to compare model output attr images to labels
    label_preprocs: a dict of {label_name: func} pairs to preprocess labels
    take_every: <int> to slice into the outputs so that there's only one per input time step

    '''
    assert all([k in target for k in label_to_attr_keys.keys()]), "label_to_attr_keys.keys() must all be in target"
    assert all([k in target for k in label_preprocs.keys()]), "label_preprocs.keys() must all be in target"

    results = {} # to store results
    labels = {k: inputs[k] for k in target}
    B,Tim = labels[label_to_attr_keys.keys()[0]].shape.as_list()[:2]

    print("target func labels", target, labels)

    # preprocess labels
    for k in labels.keys():
        func = label_preprocs.get(k, lambda x: tf.identity(x, name=("%s_identity_preproc" % k)))
        print("label, preproc", k, func)
        labels[k] = func(tf.cast(labels.pop(k), tf.float32))

    # get the valid mask
    mask = tf.cast(
        labels.get('valid', tf.ones_like(inputs[label_to_attr_keys.keys()[0]][...,0:1])),
        tf.float32
    )[...,0] # [B,Tim,Him,Wim]

    # mask out regions w invalid depth
    if max_depth is not None and 'depths' in label_preprocs.keys():
        print("masking invalid depths greater than %s" % max_depth)
        mask = mask * tf.cast(-labels['depths'][...,0] < max_depth, tf.float32)

    # denominator of MSE
    num_valid_pixels = tf.reduce_sum(mask, axis=[-2,-1])  # [B,Tim]

    # get the attributes from outputs
    decoded_images = outputs['decoded_attrs'][kwargs.get('graph_tier', -1)] # [B,Tout,P,C] with P number of sampled points
    for attr in label_to_attr_keys.values():
        decoded_im = decoded_images.pop(attr)
        _,T,P,C = decoded_im.shape.as_list()
        take_every = kwargs.get('take_every', T // Tim)
        print("take every", take_every)
        decoded_im = decoded_im[:, take_every-1::take_every]
        assert decoded_im.shape.as_list()[:2] == [B,Tim], "Must have one output per input image but Tim=%d, Tout=%d" % (Tim, T)
        assert all([P % dim == 0 for dim in im_size]), "im_size must evenly divide number of sampled points"
        decoded_im = tf.reshape(decoded_im, [B,Tim] + im_size + [C])
        decoded_images[attr] = decoded_im

    # now compute losses
    for k in label_to_attr_keys.keys():
        pred_im = decoded_images[label_to_attr_keys[k]]
        gt_im = labels[k]
        Him, Wim, Cim = gt_im.shape.as_list()[2:]
        assert Cim == pred_im.shape.as_list()[-1], "pred and gt images must have same number of channels but Cim=%d, C=%d" % (Cim, pred_im.shape.as_list()[-1])
        pred_im = tf.reshape(
            tf.image.resize_images(
                tf.reshape(pred_im, [B*Tim] + im_size + [Cim]),
                [Him, Wim]),
            [B, Tim, Him, Wim, Cim]) # now same spatial size

        attr_mse = tf.reduce_sum(tf.square(pred_im - gt_im), axis=-1)
        attr_mse = tf.reduce_sum(attr_mse * mask, axis=[2,3])
        attr_mse = attr_mse / tf.maximum(num_valid_pixels, 1.0)
        attr_mse = tf.reduce_mean(attr_mse, axis=1) # mean across time

        results[k+'_mse'] = attr_mse # [B] shape tensor of each mse per image
    return results

### AGG FUNCS ###
def just_keep_everything(val_res):
    keys = val_res[0].keys()
    return dict((k, [d[k] for d in val_res]) for k in keys)

def pickle_results_func(val_res, local_path):
    with open(local_path, 'wb') as f:
        pickle.dump(val_res, f)
    return val_res

def concatenate_examples(val_res):
    keys = val_res[0].keys()
    agg_dict = {}
    for k in keys:
        agg_dict[k] = np.concatenate([v[k] for v in val_res], axis=0)
    return agg_dict

def agg_mean_and_var(res):
    final_res = {}
    # format as dict of lists
    if isinstance(res, list):
        assert isinstance(res[0], dict), type(res[0])
        res = {k:[d[k] for d in res] for k in res[0].keys()}

    for k in res:
        assert isinstance(res[k], list), "input must be a dict of lists"
        print("length of %s result: %d" % (k, len(res[k])))
        assert isinstance(res[k][0], np.ndarray), "each entry in each val of results must be an np.ndarray"
        res_array = np.concatenate(res[k], axis=0)
        final_res[k] = np.mean(res_array, axis=0)
        final_res[k+'_var'] = np.var(res_array, axis=0)
    print(final_res)
    return final_res

def mean_res(res):
    return {k:np.mean(v) for k,v in res.items()}

def agg_mean(res):
    final_res = {}
    # format as dict of lists
    if isinstance(res, list):
        assert isinstance(res[0], dict), type(res[0])
        res = {k:[d[k] for d in res] for k in res[0].keys()}

    for k in res:
        assert isinstance(res[k], list), "input must be a dict of lists"
        print("length of %s result: %d" % (k, len(res[k])))
        assert isinstance(res[k][0], np.ndarray), "each entry in each val of results must be an np.ndarray"
        res_array = np.concatenate(res[k], axis=0)
        final_res[k] = np.mean(res_array, axis=0)
    print(final_res)
    return final_res

def agg_mean_per_time(res):
    final_res = {}
    # format as dict of lists
    if isinstance(res, list):
        assert isinstance(res[0], dict), type(res[0])
        res = {k:[d[k] for d in res] for k in res[0].keys()}
    else:
        assert isinstance(res, dict), type(res)
        for k in res.keys():
            assert isinstance(res[k], list), "results must be a list of dicts or a dict of lists"

    for k in res:
        print("length of %s result: %d" % (k, len(res[k])))
        assert isinstance(res[k][0], np.ndarray), "each entry in each val of results must be an np.ndarray"
        res_array = np.concatenate(res[k], axis=0)
        assert len(res_array.shape) == 2, res_array.shape
        T = res_array.shape[1]
        for t in range(T):
            final_res[k+'_t'+str(t)] = np.mean(res_array[:,t], axis=0)
    print(final_res)
    return final_res


### ONLINE AGG FUNCS ###
def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def append_it(x, y, step):
    if x is None:
        x = []
    x.append(y)
    return x

def append_each_val(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k,v in res.items():
        agg_res[k].append(v)
    return agg_res

def online_mean_and_var(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
        for k in res:
            agg_res[k+'_var'] = []

    for k,v in res.items():
        agg_res[k].append(np.mean(v))
        agg_res[k+'_var'].append(np.var(v))

    return agg_res

def get_aris_per_image(agg_res, res, step):

    if agg_res is None:
        agg_res = []
    pred_segments = res['pred']
    gt_segments = res['gt']

    aris = compute_aris(pred_segments, gt_segments) # [B,T] where T = num_frames
    aris = {'aris': aris}
    agg_res.append(aris)
    return agg_res

def get_mean_ari(agg_res, res, step):
    if agg_res is None:
        agg_res = []
    pred_segments = res['pred']
    gt_segments = res['gt']
    aris = compute_aris(pred_segments, gt_segments)
    ari = {'ari': np.mean(aris)}
    agg_res.append(ari)
    return agg_res

def mean_edge_accuracy_and_ari(agg_res, res, step):
    if agg_res is None:
        agg_res = []
    ari_res = get_aris_per_image(None, res, step)[0]
    new_res = {'edge_accuracy': res['edge_accuracy'], 'ari': np.mean(ari_res['aris'])}
    agg_res.append(new_res)
    return agg_res

def object_mask_and_boundary_metrics(agg_res, res, step):
    gt_objs = res['gt']
    pred_objs = res['pred']
    bde_flag = res.get('bde_flag', 0)
    ari_flag = res.get('ari_flag', 0)
    B,T = gt_objs.shape[:2]
    stride = gt_objs.shape[3] // pred_objs.shape[3]

    M = ObjectMetrics(gt_objs, pred_objs, stride=stride)
    # cluster background features
    bg_feats = res.get('background_features', np.array([0]))
    if bg_feats.sum() < 1:
        bg_feats = None

    M.cluster_unlabeled_pixels(features=bg_feats, stride=M.stride, n_clusters=res.get('background_n_clusters', 5))
    metrics = M.compute_all_metrics(examples=range(B), times=range(T),
                                    thresh=res['thresh'], stride=M.stride, connectivity=2, mode='thick',
                                    compute_BDEs=bde_flag, compute_ARIs=ari_flag, normalize=True)
    metrics = {k:metrics[k] for k in metrics.keys()
               if k in ['mIoU', 'recall', 'boundary_f_measure', 'boundary_displacement_error', 'adjusted_rand_index']}

    # split metrics and time average
    split_metrics = {}
    for k in metrics.keys():
        vshape = metrics[k].shape
        vals = np.split(metrics[k], vshape[-1], axis=-1)
        for i,val in enumerate(vals):
            split_metrics[k+('{0}').format('' if not i else i)] = val[:,:,0]

    if res.get('agg_mean', 0):
        split_metrics = {k:np.mean(split_metrics[k], axis=0, keepdims=True) for k in split_metrics.keys()}

    if agg_res is None:
        agg_res = []

    agg_res.append(split_metrics)
    return agg_res

def pred_single_tier_nodes(inputs, outputs, target, **kwargs):
    results = {}
    tier = kwargs.get('tier', 0)
    pred_times = kwargs.get('pred_times', [-1])
    # get preds
    which_nodes = [n+'_nodes' for n in kwargs.get('which_nodes', ['spatial', 'summary'])]
    nodes = [outputs[nodes][tier] for nodes in which_nodes]
    results['nodes'] = tf.concat([outputs['spatial_nodes'][tier][:,t:t+1] for t in pred_times], axis=1)
    results['summary_nodes'] = tf.concat([outputs['summary_nodes'][tier][:,t:t+1] for t in pred_times], axis=1)
    pred_particles = tf.concat([ns[...,0:3] for ns in nodes], axis=2)
    alpha_ind = kwargs.get('alpha_ind', -1)
    pred_mask = tf.nn.sigmoid(tf.concat([tf.expand_dims(ns[...,alpha_ind],-1) for ns in nodes], axis=2))

    results['pred'] = tf.concat([pred_particles[:,t:t+1] for t in pred_times], axis=1)
    results['pred_mask'] = tf.concat([pred_mask[:,t:t+1] for t in pred_times], axis=1)
    results['alphas'] = outputs['encoder_outputs'][tier]['alphas']
    results['pred_colors'] = outputs['encoder_outputs'][tier]['pred_colors']
    results['pred_particles'] = outputs['encoder_outputs'][tier]['pred_particles']
    pred_normals = outputs['encoder_outputs'][tier].get('pred_normals', None)
    if pred_normals is not None:
        results['pred_normals'] = pred_normals

    # images
    if kwargs.get('get_inputs', False):
        results['inputs'] = outputs['image_inputs']
    if kwargs.get('get_inputs_only', False):
        print(outputs['image_inputs'])
        return {'inputs': outputs['image_inputs']}

    # segment ids
    segment_ids = outputs.get('segment_ids', None)
    if segment_ids is not None:
        results['segment_ids'] = tf.concat([outputs['segment_ids'][tier][:,t:t+1] for t in pred_times], axis=1)
    summary_segment_ids = outputs.get('summary_segment_ids', None)
    if summary_segment_ids is not None:
        results['summary_segment_ids'] = tf.concat([outputs['summary_segment_ids'][tier][:,t:t+1] for t in pred_times], axis=1)

    # grouping edges
    edges = outputs.get('edges', None)
    if edges is not None:
        results['edges'] = tf.concat([edges[tier][:,t:t+1] for t in pred_times], axis=1)
    errors = outputs.get('errors', None)
    if errors is not None:
        results['errors'] = tf.concat([errors[tier][:,t:t+1] for t in pred_times], axis=1)

    # hier edges
    n2p_edges = outputs.get('n2p_edges', None)
    if n2p_edges is not None:
        results['n2p_edges'] = tf.concat([outputs['n2p_edges'][tier][:,t:t+1] for t in pred_times], axis=1)
        results['ws_edges'] = tf.concat([outputs['ws_edges'][tier][:,t:t+1] for t in pred_times], axis=1)

    # trajectories
    if kwargs.get('get_trajectories', False):
        results['trajectories'] = outputs['node_trajectories'][kwargs.get('traj_layer', tier)]

    # spatial attributes from sampled points
    if kwargs.get('get_spatial_attrs', False):
        results['spatial_attrs'] = outputs['decoded_attrs'][tier]
        results['spatial_loss'] = outputs['spatial_loss']['spatial_loss']

    if kwargs.get('get_future_attrs', False):
        results['future_attrs'] = outputs['future_attrs'][tier]
        results['future_images'] = outputs['future_images']
        results['future_segments'] = outputs['future_segments']
        results['future_nodes'] = outputs['future_nodes']
        results['node_vels'] = outputs['node_vels']

    if kwargs.get('get_shapes', False):
        results['shapes'] = outputs['shapes']
        results['shape_codes'] = outputs['shape_codes']

    # get gt
    if kwargs.get('get_gt_particles', False):
        gt_mask = tf.cast(inputs[target[1]][...,0:1], dtype=tf.float32)
        gt_full_particles = inputs[target[0]]
        gt_valid = tf.where(gt_full_particles[...,14:15] > 0.0,
                            tf.ones(gt_mask.shape, dtype=tf.float32),
                            tf.zeros(gt_mask.shape, dtype=tf.float32))
        gt_mask = gt_mask * gt_valid
        gt_particles = gt_full_particles[...,0:3]

        results['gt'] = gt_particles
        results['gt_mask'] = gt_mask
        results['gt_valid'] = gt_valid
        results['gt_full_particles'] = gt_full_particles

    # get hierarchy edges
    if kwargs.get('get_kNN', False):
        kNN = outputs['hrn_inputs']['kNN_list']
        kNN_valid = outputs['hrn_inputs']['kNN_valid_flag']
        print("kNN and kNN_valid")
        print(kNN.shape.as_list(), kNN.dtype, kNN_valid.shape.as_list(), kNN_valid.dtype)
        results['kNN'] = kNN
        results['kNN_valid'] = kNN_valid

        # last tier before objects
        if tier == (len(outputs[which_nodes[0]])-1):
            results['edges'] = outputs['encoder_outputs'][-2]['edges']
        # pred_object_masks = outputs['hierarchy']['pred_object_masks'][tier]
    if kwargs.get('get_ids', True):
        pred_object_masks = outputs['encoder_outputs'][tier]['pred_object_masks']
        results['pred_object_masks'] = pred_object_masks
        pred_object_ids = outputs['encoder_outputs'][tier]['pred_object_ids']
        results['pred_object_ids'] = pred_object_ids

    if kwargs.get('get_full_particles', False):
        assert tier == 0, "Only tier0 has full particles"
        results['full_particles_pre'] = outputs['hierarchy']['nodes'][0]
        if outputs.get('hrn_inputs', None) is not None:
            results['full_particles'] = outputs['hrn_inputs']['full_particles']
            results['full_particles_valid'] = outputs['hrn_inputs']['full_particles_valid']
        pred_object_masks = outputs['hierarchy']['pred_object_masks'][0]
        results['pred_object_masks'] = pred_object_masks
        pred_object_ids = outputs['encoder_outputs'][0]['pred_object_ids']
        results['pred_object_ids'] = pred_object_ids

    # check loss
    if kwargs.get('get_loss', False):
        loss = validate_dummy_loss(inputs, outputs, target,
                                   loss_func_kwargs=kwargs.get('loss_func_kwargs', {}))
        results['loss'] = loss['val_loss']

    return results

def build_pairwise_segment_labels(objects, inds1, inds2):
    '''
    For a ground truth segmentation image "objects" and pairs of indices inds1, inds2 (optionally broadcasted),
    determine whether each pair ((h1,w1),(h2,w2)) are in the same segment or not. Return indicator labels.

    objects: [B,T,H,W,C] <tf.uint8> an RGB segmentation map of the objects
    inds1: [B,T,P,K1,2] <tf.int32> (h,w) indices for P*K1 points into the image
    inds2: [B,T,P,K2,2] <tf.int32> (h,w) indices for P*K2 points into the image. Must have K1 == 1 or K1 == K2.
    '''
    B,T,H,W,C = objects.shape.as_list()
    _,_,P,K1,_ = inds1.shape.as_list()
    _,_,P2,K,_ = inds2.shape.as_list()
    assert P == P2, "Must use same number of points"
    assert K1 == 1 or K1 == K, "inds1 must have the same shape as inds2 or be broadcastable to it."
    if objects.dtype == tf.uint8:
        objects = object_id_hash(objects, tf.int32, val=256)
    else:
        raise NotImplementedError("object masks doesn't have a proper data format; must be 3-channel uint8.")
    assert objects.shape.as_list()[-1] == 1

    inds1 = tf.reshape(inds1, [B,T,P*K1,2])
    inds2 = tf.reshape(inds2, [B,T,P*K,2])
    inds = tf.concat([inds1, inds2], axis=2)
    segvals = get_image_values_from_indices(objects, inds) # [B,T,P*K1 + P*K,1]
    segvals1, segvals2 = tf.split(segvals, [P*K1, P*K], axis=2)
    segvals1 = tf.reshape(segvals1, [B,T,P,K1]) # [B,T,P,K1,1]
    segvals2 = tf.reshape(segvals2, [B,T,P,K])
    labels = tf.cast(tf.equal(segvals1, segvals2), tf.float32) # [B,T,P,K]

    return labels

def loss_and_in_top_k(inputs, outputs, target, logits_key, **kwargs):
    logits = tf.squeeze(outputs[logits_key], axis=1)
    labels = tf.squeeze(inputs[target], axis=1)
    return {'loss': tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels),
            'top1': tf.nn.in_top_k(logits, labels, 1),
            'top5': tf.nn.in_top_k(logits, labels, 5)}
