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
    dataset = voc_custom.VocCustomInput(tfr_dir, batch_size=2, num_exsamples=200, repeat_num=1, buffer_size=10000)

    for image, heatmap in dataset(image_augmentor_config):
        # plt.imshow(image[1])
        # plt.show()
        print(tf.shape(heatmap))

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

def gaussian_radius_tf(height, width, min_overlap=0.7):
    """
    Args:
        height, width: Both are the tensor of the same shape (N,)!
    Results:
        radius: 考虑所有框的大小，而得到的最佳半径
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = tf.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = tf.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return tf.reduce_min([r1, r2, r3])

def test_gengaussian():
    # 这是值得读的一段代码
    # 在不同位置生成Gaussian 分布
    pshape = np.array([256, 256])
    # 下面三者，0维的大小对应
    center = tf.cast(np.array([[100, 100], [200, 200]]), dtype=tf.int64)
    h = tf.cast(np.array([13., 11.]), dtype=tf.float32)
    w = tf.cast(np.array([17., 9.]), dtype=tf.float32)

    sigma = gaussian_radius_tf(h, w, 0.7)

    c_y = tf.cast(center[:, 0], dtype=tf.float32)
    c_x = tf.cast(center[:, 1], dtype=tf.float32)
    # 注意下面的一步，其用意在于后面c_y-mesh_y之类的运算
    c_y = tf.reshape(c_y, [-1, 1, 1])
    c_x = tf.reshape(c_x, [-1, 1, 1])
    # 中心的个数
    num_g = tf.shape(center)[0]

    y_range = tf.range(0, pshape[0], dtype=tf.float32)
    x_range = tf.range(0, pshape[1], dtype=tf.float32)
    [mesh_x, mesh_y] = tf.meshgrid(x_range, y_range)
    mesh_x = tf.expand_dims(mesh_x, 0)
    mesh_x = tf.tile(mesh_x, [num_g, 1, 1])
    mesh_y = tf.expand_dims(mesh_y, 0)
    mesh_y = tf.tile(mesh_y, [num_g, 1, 1])
    heatmap = tf.exp(-((c_y-mesh_y)**2+(c_x-mesh_x)**2)/(2*sigma**2))

    # 合成
    heatmap = tf.reduce_max(heatmap, axis=0)
    print(heatmap[center[0, 0], center[0, 1]])
    print(heatmap[center[1, 0], center[1, 1]])
    print(heatmap[101, 101])

    heatmap = tf.expand_dims(heatmap, axis=-1)
    all = []
    for i in range(5):
        all.append(heatmap)
    heatmap = tf.concat(all, axis=-1)
    print(heatmap[201, 201, :])


if __name__ == '__main__':
    # about_dataset_voc()
    # repair_data("./data_voc/Annotations/")
    # tf.executing_eagerly()
    # make_voc_custom_dataset()
    about_dataset_voc_custom()
    # test_gather()
    # test_meshgrid()
    # test_gengaussian()
