import numpy as np
import tensorflow.compat.v1 as tf
import os

class DataProvider(object):

    def __init__(
            self,
            data_paths,
            sequence_len=1,
            file_pattern="*.tfrecords",
            train_len=None,
            val_len=None,
            **kwargs
    ):
        self.data_paths = data_paths
        self.sequence_len = sequence_len
        self.is_training = None
        self.file_pattern = file_pattern
        self.TRAIN_LEN = train_len
        self.VAL_LEN = val_len

    @staticmethod
    def get_data_params(batch_size, sequence_len, **kwargs):
        raise NotImplementedError("You must overwrite the get_data_params staticmethod for the data provider class!")

    def get_tfr_filenames(self, folder_name):
        tfrecord_pattern = os.path.join(folder_name, self.file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return datasource

    def preprocessing(self, image_string):
        raise NotImplementedError("Preprocessing not implemented")

    def input_fn(self, batch_size, train, **kwargs):
        raise NotImplementedError("You must overwrite the input_fn method for the %s data provider class!" % type(self).__name__)
