from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import os, sys, copy
import numpy as np
import pdb
from .base import DataProvider
from .utils import *
from .kinetics_transform import video_transform_color
import os
import os.path
from collections import namedtuple
import time
import random
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose

VideoRecord = namedtuple('VideoRecord', ['path', 'num_frames', 'label'])
ClipRecord = namedtuple('ClipRecord', ['path', 'start_frame_no', 'num_frames', 'label'])
VideoCollageRecord = namedtuple('VideoCollageRecord', ['path', 'num_frames', 'size', 'label'])

META_PATH = '/mnt/fs3/chengxuz/kinetics/pt_meta'

class Kinetics(DataProvider):
    """
    Class where data provider for Kinetics will be built
    """
    DATA_PATH = '/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted'  # node07    

    def __init__(self,
                 data_paths,
                 prep_type='resnet',
                 crop_size=256,
                 smallest_side=256,
                 resize=None,
                 images_key='images',
                 labels_key='labels',
                 temporal=True,
                 do_color_normalize=False,
                 q_cap=51200,
                 **kwargs):

        self.cfg = self.dataset_config(root_meta=META_PATH, root_data=data_paths)
        self.images_key = images_key
        self.labels_key = labels_key
        self.crop_size = crop_size
        self.smallest_side = smallest_side

    @staticmethod
    def dataset_config(root_meta, root_data):
        """Return the split information."""
        file_categories = os.path.join(root_meta, 'categories.txt')
        file_imglist_train = os.path.join(root_meta, 'train_frameno_new.txt')
        file_imglist_val = os.path.join(root_meta, 'val_frameno_new.txt')
        prefix = '{:06d}.jpg'

        with open(file_categories) as f:
            categories = [line.rstrip() for line in f.readlines()]

        return {
            'categories': categories,
            'train_metafile': file_imglist_train,
            'val_metafile': file_imglist_val,
            'root': root_data,
            'prefix': prefix
        }

    @staticmethod
    def get_data_params(batch_size, sequence_len=None, dataprefix=DATA_PATH, **kwargs):
        data_params = copy.deepcopy(kwargs)
        data_params['data_paths'] = dataprefix
        data_params['temporal'] = (sequence_len is not None)
        return copy.deepcopy(data_params), copy.deepcopy(data_params)

    def input_fn(self, batch_size, train, **kwargs):
        """
        Build the dataset, dataloader, get the elements
        """
        transform = video_transform_color()

        # build dataset
        dataset = VideoDataset(
            self.cfg['root'],
            self.cfg['train_metafile'] if train else self.cfg['val_metafile'],
            trn_style=True,
            transform=transform,
            frame_start='RANDOM',
            trn_num_frames=4)

        # build dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=30, pin_memory=False,
            worker_init_fn=lambda x: np.random.seed(x))

        # build data enumerator
        data_enumerator = enumerate(dataloader)

        # get next element
        _, (input, target, index) = data_enumerator.next()        

        # input image is of shape [B, T, H, W, C]
        # label is of shape [B, ]
        next_element = {
            self.images_key: tf.convert_to_tensor(input.numpy()),
            self.labels_key: tf.convert_to_tensor(target.numpy()),
        }

        return next_element


class VideoDataset(data.Dataset):
    '''
    Build pytorch data provider for loading frames from videos
    Args:
        root (str):
            Path to the folder including all jpgs
        metafile (str):
            Path to the metafiles
        frame_interval (int):
            number of frames to skip between two sampled frames, None means
            interval will be computed so that the frames subsampled cover the
            whole video
        frame_start (str):
            Methods of determining which frame to start, RANDOM means randomly
            choosing the starting index, None means the middle of valid range
    '''

    MIN_NUM_FRAMES = 3

    def __init__(
            self, root, metafile,
            num_frames=8, frame_interval=None, frame_start=None,
            file_tmpl='{:06d}.jpg', transform=None, sample_groups=1,
            bin_interval=None,
            trn_style=False, trn_num_frames=8,
            part_vd=None,
            clip_meta=False,
            drop_index=False):

        self.root = root
        self.drop_index = drop_index
        self.metafile = metafile
        self.file_tmpl = file_tmpl
        self.transform = transform

        if transform is None:
            # if transform is None, self.transform is an identity map
            self.transform = lambda x: x

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_start = frame_start
        self.sample_groups = sample_groups
        self.bin_interval = bin_interval
        self.trn_style = trn_style
        self.trn_num_frames = trn_num_frames
        self.part_vd = part_vd
        self.clip_meta = clip_meta
        if 'infant' in self.metafile:
            self.clip_meta = True

        self._parse_list()

    def _load_image(self, directory, idx):
        tmpl = os.path.join(self.root, directory, self.file_tmpl)

        try:
            return Image.open(tmpl.format(idx)).convert('RGB')
        except Exception:
            print('error loading image: {}'.format(tmpl.format(idx)))
            return Image.open(tmpl.format(1)).convert('RGB')

    def __get_interval_valid_range(self, rec_no_frames):
        if self.frame_interval is None:
            interval = rec_no_frames / float(self.num_frames)
        else:
            interval = self.frame_interval
        valid_sta_range = rec_no_frames - (self.num_frames - 1) * interval
        return interval, valid_sta_range

    def _build_bins_for_vds(self):
        self.video_bins = []
        self.bin_curr_idx = []
        self.video_index_offset = []
        curr_index_offset = 0

        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            _, valid_sta_range = self.__get_interval_valid_range(
                rec_no_frames)
            curr_num_bins = np.ceil(valid_sta_range * 1.0 / self.bin_interval)
            curr_num_bins = int(curr_num_bins)
            curr_bins = [
                (_idx,
                 (self.bin_interval * _idx,
                  min(self.bin_interval * (_idx + 1),
                      valid_sta_range)))
                for _idx in range(curr_num_bins)]

            self.video_bins.append(curr_bins)
            self.bin_curr_idx.append(0)
            self.video_index_offset.append(curr_index_offset)
            np.random.shuffle(self.video_bins[-1])

            curr_index_offset += curr_num_bins
        return curr_index_offset

    def _build_trn_bins(self):
        num_bins = self.trn_num_frames
        half_sec_frames = 12
        all_vds_bin_sta_end = []
        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            frame_each_bin = min(half_sec_frames, rec_no_frames // num_bins)

            if frame_each_bin == 0:
                all_vds_bin_sta_end.append([])
                continue

            curr_bin_sta_end = []
            for curr_sta in range(0, rec_no_frames, frame_each_bin):
                curr_bin_sta_end.append(
                    (curr_sta,
                     min(curr_sta + frame_each_bin, rec_no_frames)))
            assert len(curr_bin_sta_end) >= num_bins
            all_vds_bin_sta_end.append(curr_bin_sta_end)

        self.all_vds_bin_sta_end = all_vds_bin_sta_end

    def _parse_list(self):
        # check the frame number is >= MIN_NUM_FRAMES
        # usualy it is [video_id, num_frames, class_idx]
        with open(self.metafile) as f:
            len_frame_idx = 1 if not self.clip_meta else 2
            lines = [x.strip().split(' ') for x in f]
            lines = [line for line in lines
                     if int(line[len_frame_idx]) >= self.MIN_NUM_FRAMES]

        record_type = VideoRecord
        if self.clip_meta:
            record_type = ClipRecord
        self.video_list = [record_type(*v) for v in lines]
        if self.part_vd is not None:
            np.random.seed(0)
            now_len = len(self.video_list)
            chosen_indx = sorted(np.random.choice(
                range(now_len), int(now_len * self.part_vd)))
            self.video_list = [self.video_list[_tmp_idx]
                               for _tmp_idx in chosen_indx]

        print('Number of videos: {}'.format(len(self.video_list)))
        if self.bin_interval is not None:
            num_bins = self._build_bins_for_vds()
            print('Number of bins: {}'.format(num_bins))

        if self.trn_style:
            self._build_trn_bins()

    def _get_indices(self, record):
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        all_offsets = None
        start_interval = valid_sta_range / (1.0 + self.sample_groups)
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
            offsets = np.array([int(sta_idx + interval * x)
                                for x in range(self.num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_binned_indices(self, index, record):
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        _bin_curr_idx = self.bin_curr_idx[index]
        _idx, (_sta_random,
               _end_random) = self.video_bins[index][_bin_curr_idx]
        assert self.frame_start == 'RANDOM', "Binned only supports random!"

        sta_idx = np.random.randint(_sta_random, _end_random)
        offsets = np.array([int(sta_idx + interval * x)
                            for x in range(self.num_frames)])
        self.bin_curr_idx[index] += 1
        if self.bin_curr_idx[index] == len(self.video_bins[index]):
            self.bin_curr_idx[index] = 0
            np.random.shuffle(self.video_bins[index])
        return offsets + 1, _idx + self.video_index_offset[index]

    def _get_valid_video(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(
                os.path.join(self.root, record.path, self.file_tmpl.format(1))):
            print(
                os.path.join(
                    self.root,
                    record.path,
                    self.file_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if self.frame_interval is not None or self.trn_style:
            needed_frames = self.get_needed_frames()
            while int(record.num_frames) <= needed_frames:
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        return record, index

    def get_needed_frames(self):
        if not self.trn_style:
            needed_frames = self.num_frames * self.frame_interval
        else:
            needed_frames = self.trn_num_frames
        return needed_frames

    def _get_TRN_style_indices(self, index, record):
        curr_bin = self.all_vds_bin_sta_end[index]

        valid_sta_range = len(curr_bin) - self.trn_num_frames + 1
        all_offsets = None
        start_interval = int(valid_sta_range / (1.0 + self.sample_groups))
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
                offsets = np.array([int(np.mean(curr_bin[sta_idx + x]))
                                    for x in range(self.trn_num_frames)])
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
                offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])
                                    for x in range(self.trn_num_frames)])

            offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])
                                for x in range(self.trn_num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_indices_and_instance_index(self, index, record):
        if self.bin_interval is None:
            if not self.trn_style:
                indices = self._get_indices(record)
            else:
                indices = self._get_TRN_style_indices(index, record)
            vd_instance_index = index
        else:
            indices, vd_instance_index = self._get_binned_indices(
                index, record)
        return indices, vd_instance_index

    def __getitem__(self, index):
        record, index = self._get_valid_video(index)

        indices, vd_instance_index = self._get_indices_and_instance_index(
            index, record)

        idx_offset = 0
        if self.clip_meta:
            idx_offset = int(record.start_frame_no)
        frames = self.transform([self._load_image(record.path, int(idx) + idx_offset)
                                 for idx in indices])
        if not self.drop_index:
            return frames, int(record.label), vd_instance_index
        else:
            return frames, int(record.label)

    def __len__(self):
        return len(self.video_list)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    BATCH_SIZE = 4
    TRAIN = True
    train_data, val_data = Kinetics.get_data_params(BATCH_SIZE)
    data_provider = Kinetics(**(train_data if TRAIN else val_data))
    func = data_provider.input_fn
    data = func(BATCH_SIZE, TRAIN)
    print("data", data)
