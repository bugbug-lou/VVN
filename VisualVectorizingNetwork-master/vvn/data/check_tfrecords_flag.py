import numpy as np
import tensorflow as tf

flag = 'is_moving'
#flag = 'is_object_in_view'
dataset = 'ball_hits_primitive2_experiment'
f = tf.python_io.tf_record_iterator(path='/mnt/fs1/datasets/'+ str(dataset) + \
        '/new_tfdata/' + flag + '/' + '1-0-3.tfrecords')
# f = tf.python_io.tf_record_iterator(path='/mnt/fs4/cfan/tdw-agents/data/'+ str(dataset) + \
#         '/new_tfdata/' + flag + '/' + '0-0-3.tfrecords')
datum = tf.train.Example()

counter = 0
y = []
try:
    while(True):
        data = f.next()
        datum.ParseFromString(data)
        feat = datum.features.feature[flag].bytes_list.value[0]
        x = np.reshape(np.fromstring(feat, dtype=np.float32), [-1])
        print(counter, x[:])
        y.append(x)
        counter += 1
        if counter == 256:
            break
except StopIteration:
    print('%d actions found.' % sum(y))
