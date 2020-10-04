from __future__ import print_function
import argparse
import glob
import os
import ast
import copy
import numpy as np
# import debug_init_paths
import tensorflow as tf
import torchvision.transforms as T
from PIL import Image
from .base import DataProvider
from .utils import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess(x):
    '''cannot find corresponding normalization function for tensorflow, but this is a trivial neglect'''
    return x

def Resize(x, size, interp = Image.BILINEAR):

    if isinstance(size, tuple):
        H, W = size
        tsize = (W, H)
    else:
        tsize = (size, size)
        return x.resize(tsize, interp)

class ShapeStacks(DataProvider):
    DATA_PATH = 'C:\shapestacks/frc_35/'
    COMMON_LIST = 'C:\shapestacks/splits/'
    def __init__(self, data_paths, common_list, dataset_names, dt, radius, mod, txt, Obj_num,
                 sequence_len, normalize_images=True, max_samples=None):
        super().__init__(data_paths)

        self.RW = self.RH = self.W = self.H = 224
        self.orig_W = self.orig_H = 224
        self.box_rad = radius

        self.common_list = common_list
        self.ext = '.jpg'
        self.max_samples = max_samples
        self.dataset_names = dataset_names
        self.dt = dt
        self.num_obj = 0
        self.modality = mod
        self.O = Obj_num
        self.sequence_len = sequence_len

        self.common_list = common_list
        self.data_paths = data_paths
        self.txt = txt

        list_path = self.common_list + self.txt
        with open(list_path) as fp:
            self.index_list = [line.split()[0] for line in fp]
        self.num_examples = len(self.index_list)
        self.roidb = self.parse_gt_roidb()
        eg_path = glob.glob(os.path.join(self.data_paths, self.index_list[0], self.modality + '*' + self.ext))[0]
        self.image_pref = '-'.join(os.path.basename(eg_path).split('-')[0:-1])

    @staticmethod
    def get_data_params(batch_size, dataset_names, sequence_len=1, dataprefix=DATA_PATH, common_list = COMMON_LIST, **kwargs):
        data_params = copy.deepcopy(kwargs)
        data_params['data_paths'] = dataprefix
        data_params['sequence_len'] = (sequence_len is not None)
        data_params['common_list'] = common_list
        data_params['dataset_names'] = dataset_names
        if dataset_names in ['ss3']:
            data_params['common_list'] = data_params['common_list'] + '/env_ccs+blocks-hard+easy-h=3-vcom=1+2+3-vpsf=0/'
            data_params['Obj_num'] = 3
        else:
            num = int(dataset_names[2])
            data_params['common_list'] = data_params['common_list']  + '/env_ccs+blocks-hard+easy-h=%d-vcom=1+2+3+4+5+6-vpsf=0/' % num
            data_params['Obj_num'] = num
        tr_data_params = copy.deepcopy(data_params)
        val_data_params = copy.deepcopy(data_params)
        tr_data_params['txt'] = 'train.txt'
        val_data_params['txt'] = 'eval.txt'

        return tr_data_params, val_data_params


    def transform(self, x):
        #x = Resize(x, (self.H, self.W))
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, dtype = float)
        x = imagenet_preprocess(x)
        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, 0)

        return x

    def obj_transform(self, x):
        #x = Resize(x, (self.RH, self.RW))
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, dtype = float)
        x = imagenet_preprocess(x)
        x = tf.expand_dims(x, 0)
        x = tf.expand_dims(x, 0)

        return x


    def parse_gt_roidb(self):
        roidb = {}
        for index in self.index_list:
            gt_path = os.path.join(self.data_paths, index, 'cam_1.npy')
            bbox = np.load(gt_path)  ## 32, 3, 2 in (0, 224) coor
            roidb[index] = bbox
            self.num_obj = bbox.shape[1]
        return roidb

    def __len__(self):
        num = len(self.index_list)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num

    def generate(self, index, batch):
        """
        :return: src, dst. each is a list of object
        - 'image': FloatTensor of shape (dt, C, H, W). resize and normalize to faster-rcnn
        - 'crop': (O, C, RH, RW) RH >= 224
        - 'bbox': (O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (T, 3)
        - 'index': (dt,)
        """

        vid_point = []
        for dt in range(self.dt):
            this_index = self.get_index_after(self.index_list[index], dt)
            vid_obj = {}
            norm_bbox = self.roidb[self.index_list[index]][dt]  # (O, 2)
            bboxes = np.vstack((norm_bbox[:, 0] * self.orig_W, norm_bbox[:, 1] * self.orig_H)).T
            image, crops = self._read_image(this_index, bboxes)

            trip = self._build_graph(this_index)
            vid_obj['index'] = this_index
            vid_obj['image'] = image
            vid_obj['crop'] = crops
            vid_obj['bbox'] = tf.cast(norm_bbox, dtype = float)
            vid_obj['trip'] = tf.cast(trip, dtype = float)
            valid = np.arange(3, dtype=np.int64)
            vid_obj['info'] = (self.orig_W, self.orig_H, valid)
            vid_obj['valid'] = tf.cast(valid, dtype = float)

            vid_point.append(vid_obj)

        vid_dic = {}
        key_set = ['index', 'image', 'crop', 'bbox', 'trip', 'info', 'valid']
        for key in key_set:
            vid_dic[key] = []
            for i in range(len(vid_point)):
                k = vid_point[i][key]
                vid_dic[key].append(k)
        return vid_dic

    def get_index_after(self, index, dt):
        return os.path.join(index, self.image_pref + '-%02d' % dt)

    def _read_image(self, index, bboxes):
        image_path = os.path.join(self.data_paths, index) + self.ext
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                dst_image = self.transform(image.convert('RGB'))
                crops = self._crop_image(index, image, bboxes)
        return dst_image, crops
    

    def _crop_image(self, index, image, box_center):
        crop_obj = []
        x1 = box_center[:, 0] - self.box_rad
        y1 = box_center[:, 1] - self.box_rad
        x2 = box_center[:, 0] + self.box_rad
        y2 = box_center[:, 1] + self.box_rad

        bbox = np.vstack((x1, y1, x2, y2)).transpose()
        for d in range(len(box_center)):
            crp = image.crop(bbox[d]).convert('RGB')
            crp = self.transform(crp)
            crop_obj.append(crp)
        crop_obj = tf.stack(crop_obj)
        return crop_obj

    def _build_graph(self, index):
        all_trip = np.zeros([0, 3], dtype=np.float32)
        for i in range(self.num_obj):
            for j in range(self.num_obj):
                trip = [i, 0, j]
                all_trip = np.vstack((all_trip, trip))
        return tf.cast(all_trip, dtype = float)

    def input_fn(self, batch_size, train, **kwargs):
        '''return data of the form:
        - 'image': FloatTensor of shape (B, dt, C, H, W). resize and normalize to faster-rcnn
        - 'crop': (B, O, C, RH, RW) RH >= 224
        - 'bbox': (B, O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (B, T, 3)
        - 'index': (dt,)
        '''
        self.training = train
        input_dict = {}
        key_set = ['image', 'crop', 'bbox', 'trip']
        for key in key_set:
            input_dict[key] = []
        for index in range(self.num_examples):
            for key in key_set:
                t = self.generate(index)[key]
                input_dict[key].append(t)

        num = self.num_examples

        def input_dict_gen():
            for i in range(num):
                ls = {}
                for key, val in input_dict.items():
                    ls[key] = val[i]
                yield ls

        data = tf.data.Dataset.from_generator(
            input_dict_gen,
            output_types={k: tf.float32 for k in input_dict},
            output_shapes={'image': (self.H, self.W, 3), 'crop': (self.RH, self.RW, 3), 'bbox': (self.O, 4),
                           'trip': (int(self.O ** 2), 3)})
        data = data.batch(batch_size)
        if self.training:
            data = data.shuffle(num)

        iter = tf.compat.v1.data.make_one_shot_iterator(data)
        inputs = iter.get_next()

        #data = tf.data.Dataset.from_tensor_slices(input_dict)

        return inputs


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    BATCH_SIZE = 4
    TRAIN = False
    DATASET = 'ss3'
    DT = 16
    RAD = 35
    modality = 'rgb'

    train_data, val_data = ShapeStacks.get_data_params(BATCH_SIZE, dataset_names=DATASET)
    data_provider = ShapeStacks(dt=DT, radius = RAD, mod = modality,**(train_data if TRAIN else val_data))
    func = data_provider.input_fn
    inputs = func(BATCH_SIZE, TRAIN)
    print("data", inputs)

    with tf.compat.v1.Session() as sess:
        for i in range(10):
            print(sess.run(inputs))


