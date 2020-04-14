import os
# import hashlib

# import multiprocessing
# import numpy as np
# import skimage.io as io

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from lxml import etree
from tensorgroup.models.dataset import mode_keys as ModeKey

KA = tf.keras.applications
KL = tf.keras.layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


image_shape = (384, 384, 3)

lr = 0.001
batch_size = 15
buffer_size = 256
epochs = 160
reduce_lr_epoch = []

MAX_OBJECTS = 100

config = {
    'mode': 'train',                                       # 'train', 'test'
    'input_size': 384,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.1,
    'top_k_results_output': MAX_OBJECTS,

}

image_augmentor_config = {
    'data_format': 'channels_last',
    'network_input_shape': [512, 512],                           # Must match the network's input_shape!
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'color_jitter_prob': 0.5,
    'pad_truth_to': MAX_OBJECTS,                                   # Must match the maximal objects!
}

def about_dataset_voc():
    from tensorgroup.models.dataset.voc import voc
    from tensorgroup.models.dataset import mode_keys as MK

    dataset = voc.VocInput(MK.TRAIN, batch_size=2, num_exsamples=4)

    for image, gt in dataset(image_augmentor_config):
        plt.imshow(image[1])
        plt.show()
        print(gt.numpy)

def make_voc_custom_dataset():
    from tensorgroup.models.dataset.voc import voc_custom
    ann_dir, img_dir, tfr_dir = "./data_voc/Annotations", "./data_voc/Annotations", "./data_voc/tf_records"
    voc_custom.dataset2tfrecord(ann_dir, img_dir, tfr_dir, ModeKey.TRAIN)

def about_dataset_voc_custom():
    from tensorgroup.models.dataset.voc import voc_custom
    tfr_dir = "./data_voc/tf_records"
    dataset = voc_custom.VocCustomInput(tfr_dir, batch_size=2, num_exsamples=100, repeat_num=1, buffer_size=10000)

    for image, heatmap in dataset(image_augmentor_config):
        # plt.imshow(image[1])
        # plt.show()
        print(heatmap)

def repair_data(ann_dir):
    xmllist = tf.io.gfile.glob(os.path.join(ann_dir, '*.xml'))
    for xmlpath in xmllist:
        xml = etree.parse(xmlpath)
        root = xml.getroot()
        image_name = root.find('filename')
        path = root.find('path')
        path.text = image_name.text
        xml.write(xmlpath)

def test_gather():
    # 琢磨一下 tf.gather_nd
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    print("a shape is {}".format(a.shape))
    # b = np.array([[[0, 1]], [[2, 3]]])
    b = np.array([[0], [0], [3], [3]])
    print("b shape is {}".format(b.shape))
    print(tf.gather_nd(a, b, batch_dims=1))
    print(tf.gather_nd(a, b, batch_dims=0))
    print("-------------------------------")
    b = np.array([[0, 1], [2, 3]])
    print("b shape is {}".format(b.shape))
    # print(tf.gather_nd(a, b, batch_dims=1))
    print(tf.gather_nd(a, b, batch_dims=0))

def test_meshgrid():
    # 和np.ogrid似乎对应
    m, n = 4, 3
    y = tf.range(-m, m+1, dtype=tf.float32)
    x = tf.range(-n, n+1, dtype=tf.float32)
    [n_x, m_y] = tf.meshgrid(x, y)
    print(n_x)
    print(m_y)
    h = gaussian2D_tf(np.array([9, 7]))
    print(h)

def gaussian2D_tf(shape, sigma=1):
    m, n = shape[0], shape[1]
    m = tf.cast((m-1.)/2, dtype=tf.float32)
    n = tf.cast((n-1.)/2, dtype=tf.float32)
    # m, n = shape[0], shape[1]
    # m, n = (m - 1.0)/2, (n - 1.0)/2
    y = tf.range(-m, m+1, dtype=tf.float32)
    x = tf.range(-n, n+1, dtype=tf.float32)
    [n_x, m_y] = tf.meshgrid(x, y)

    h = tf.exp(-(n_x * n_x + m_y * m_y) / (2 * sigma * sigma))
    # h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


if __name__ == '__main__':
    # about_dataset_voc()
    # repair_data("./data_voc/Annotations/")
    # tf.executing_eagerly()
    # make_voc_custom_dataset()
    # about_dataset_voc_custom()
    # test_gather()
    test_meshgrid()
