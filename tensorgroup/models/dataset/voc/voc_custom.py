from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from lxml import etree
import os
import numpy as np
import math
import sys

from typing import Text, Optional
from tensorgroup.models.dataset.image_augmentor import image_augmentor
from tensorgroup.models.dataset import mode_keys as ModeKey

"""
为提高效率，先将目录文件式的数据转换成tfrecord格式的数据。
"""

voc_custom_classes = {
    'single_white': 0,
    'double_red': 1,
    'word': 2
}

TFR_PATTERN = '{}_*.tfrecords'


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=values))


def xml_to_example(xmlpath, imgpath):
    xml = etree.parse(xmlpath)
    root = xml.getroot()
    imgname = root.find('filename').text
    imgname = os.path.join(imgpath, imgname)
    image = tf.io.gfile.GFile(imgname, 'rb').read()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)
    shape = np.asarray([height, width, depth], np.int32)
    xpath = xml.xpath('//object')
    ground_truth = np.zeros([len(xpath), 5], np.float32)
    for i in range(len(xpath)):
        obj = xpath[i]
        classid = voc_custom_classes[obj.find('name').text]
        bndbox = obj.find('bndbox')
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        # Normalized the objects cordinations
        ground_truth[i, :] = np.asarray([ymin/height,
                                         ymax/height,
                                         xmin/width,
                                         xmax/width,
                                         classid],
                                        np.float32)
    features = {
        'image': bytes_feature(image),
        'shape': bytes_feature(shape.tobytes()),
        'ground_truth': bytes_feature(ground_truth.tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(
        feature=features))
    return example


def dataset2tfrecord(xml_dir, img_dir, output_dir, name, total_shards=2):
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    tf.io.gfile.mkdir(output_dir)
    outputfiles = []
    xmllist = tf.io.gfile.glob(os.path.join(xml_dir, '*.xml'))
    num_per_shard = int(math.ceil(len(xmllist)) / float(total_shards))
    for shard_id in range(total_shards):
        outputname = '%s_%05d-of-%05d.tfrecords' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)
        with tf.io.TFRecordWriter(outputname) as tf_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(xmllist))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % (
                    i+1, len(xmllist), shard_id+1, total_shards))
                sys.stdout.flush()
                example = xml_to_example(xmllist[i], img_dir)
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles


class VocCustomInput:
    """
    定义Voc类训练库的什么呢？
    """

    def __init__(self,
                 tfrecords_dir,
                 mode: Text = ModeKey.TRAIN,
                 batch_size: Optional[int] = -1,
                 num_exsamples: Optional[int] = -1,
                 repeat_num: Optional[int] = -1,
                 buffer_size: Optional[int] = -1,
                 max_objects: Optional[int] = 100):
        assert mode is not None
        self._tfrecords_dir = tfrecords_dir
        self._mode = mode
        self._batch_size = batch_size
        self._num_examples = num_exsamples
        self._repeat_num = repeat_num
        self._buffer_size = buffer_size
        #
        self._num_classes = len(voc_custom_classes)
        self._max_objects = max_objects

    def __call__(self, image_augmentor_config):
        tfrecords = tf.io.gfile.glob(os.path.join(self._tfrecords_dir, TFR_PATTERN.format(self._mode)))
        print(tfrecords)
        dataset = tf.data.TFRecordDataset(tfrecords)
        decoder = Decoder(ImageNormalizer())
        inputs_def = DefineInputs(image_augmentor_config,
                                  num_classes=self._num_classes,
                                  max_objects=self._max_objects)
        dataset = dataset.map(decoder).map(inputs_def)  # 定义输入的内容、格式！ dataset = (image, heatmap)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)
        if self._repeat_num > 0:
            dataset = dataset.repeat(self._repeat_num)
        if self._buffer_size > 0:
            dataset = dataset.shuffle(self._buffer_size)
        if self._batch_size > 0:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)

        return dataset

class Decoder:
    def __init__(self, image_normalizer):
        self._image_normalizer = image_normalizer

    def __call__(self, tfrecord):
        features = tf.io.parse_single_example(tfrecord, features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([], tf.string),
            'ground_truth': tf.io.FixedLenFeature([], tf.string)
        })
        shape = tf.io.decode_raw(features['shape'], tf.int32)
        ground_truth = tf.io.decode_raw(features['ground_truth'], tf.float32)
        shape = tf.reshape(shape, [3])
        ground_truth = tf.reshape(ground_truth, [-1, 5])
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.reshape(image, shape)
        if self._image_normalizer is not None:
            image = self._image_normalizer(image)
        return image, ground_truth

class DefineInputs:
    """
    """

    def __init__(self, config, num_classes, max_objects):
        """ 指定网络需要的输入格式。
        Args:
            config:
              {'data_format': 'channels_last',
               'network_input_shape': [512, 512],                           # Must match the network's input_shape!
               'flip_prob': [0., 0.5],
               'fill_mode': 'BILINEAR',
               'color_jitter_prob': 0.5,
               'pad_truth_to': 100,                                   # Must match the maximal objects!
              }
            num_classes:
        """
        self._config = config
        self._num_classes = num_classes
        self._max_objects = max_objects

    def __call__(self, image, ground_truth):
        """将features转换成模型需要的形式
        Args:
            image: Normalized image.
            ground_truth: Normalized Ground truth!
        Returns:
            images:
            center_keypoint_heatmap:
            center_offset:
            regression_shape:

        """

        image, ground_truth = image_augmentor(image=image,
                                              ground_truth=ground_truth,
                                              **self._config
                                              )
        # ground_truth: [y_center, x_center, height, width, classid]

        heatmap, mask = self._def_inputs(image, ground_truth)
        return image, heatmap, mask
        # return image, ground_truth

    def _def_inputs(self, image, ground_truth):
        """生成网络所需要的输入。
        Args:
            image: 已经调整成了self._config['network_input_shape']
            ground_truth: y_x_h_w_class格式，且归一化了, no_padding。
        Results:
            center_keypoint_heatmap
            center_offset_reg
            center_shape_reg
        """
        center_keypoint_heatmap = self._gen_center_keypoint_heatmap(ground_truth)
        return center_keypoint_heatmap

    def _gen_center_keypoint_heatmap(self, ground_truth):
        # objects_num = tf.shape(ground_truth)[0]
        # center_keypoint_index = tf.zeros((self._max_objects), tf.dtypes.int64)
        # network_input_shape = self._config['network_input_shape']
        # (i_h, i_w) = network_input_shape
        # (f_h, f_w) = (int(i_h/4), int(i_w/4))
        # network_featuremap_shape = (f_h, f_w, self._num_classes)
        # center_keypoint_heatmap = tf.zeros(network_featuremap_shape, dtype=tf.float32)
        # center_offset_reg = tf.zeros((self._max_objects, 2), dtype=tf.float32)
        # center_keypoint_index = tf.zeros((self._max_objects), dtype=tf.int64)

        # print(objects_num)

        # for k in range(objects_num):
        #     (y, x, h, w, class_id) = ground_truth[k]     # Nomalized coordinates!
        #     (y, x, h, w) = (y*f_h, x*f_w, h*f_h, w*f_w)  # featuremap中的坐标
        #     (y_int, x_int, h_int, w_int) = int((y, x, h, w)+0.5)  # featuremap中的离散坐标
        #     (y_off, x_off, h_off, w_off) = (y, x, h, w) - (y_int, x_int, h_int, w_int)  # 离散坐标和原始坐标的偏差
        #     center_offset_reg[k] = [y_off, x_off]
        #     center_keypoint_index[k] = y_int*f_w+x_int

        #     radius = gaussian_radius((h_int, w_int))
        #     radius = max(0, int(radius))
        #     draw_gaussian(center_keypoint_heatmap, layer=int(class_id), center=[y_int, x_int], sigma=radius)

        #     pass
        # return center_keypoint_heatmap

        network_input_shape = self._config['network_input_shape']
        (i_h, i_w) = network_input_shape
        (f_h, f_w) = (int(i_h/4), int(i_w/4))

        objects_num = tf.argmin(ground_truth, axis=0)
        objects_num = objects_num[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, objects_num, dtype=tf.int32))

        # ground_truth的shape应为(objects_num, 5)
        c_y = ground_truth[..., 0] * f_h
        c_x = ground_truth[..., 1] * f_w
        c_h = ground_truth[..., 2] * f_h
        c_w = ground_truth[..., 3] * f_w
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)

        center = tf.concat([tf.expand_dims(c_y, axis=-1), tf.expand_dims(c_x, axis=-1)], axis=-1)
        c_y = tf.floor(c_y)
        c_x = tf.floor(c_x)
        center_round = tf.floor(center)
        # center_offset_reg = center - center_round
        center_round = tf.cast(center_round, dtype=tf.int64)

        center_keypoint_heatmap = gaussian2D_tf_at_any_point(objects_num, c_y, c_x, c_h, c_w, f_h, f_w)
        zero_like_heatmap = tf.expand_dims(tf.zeros([f_h, f_w], dtype=tf.float32), axis=-1)
        all_class_heatmap = []
        all_class_mask = []
        for i in range(self._num_classes):
            is_class_i = tf.equal(class_id, i)

            class_i_heatmap = tf.boolean_mask(center_keypoint_heatmap, is_class_i, axis=0)
            class_i_heatmap = tf.cond(
                tf.equal(tf.shape(class_i_heatmap)[0], 0),
                lambda: zero_like_heatmap,
                lambda: tf.expand_dims(tf.reduce_max(class_i_heatmap, axis=0), axis=-1)
            )
            all_class_heatmap.append(class_i_heatmap)

            class_i_center = tf.boolean_mask(center_round, is_class_i, axis=0)
            class_i_mask = tf.cond(
                tf.equal(tf.shape(class_i_center)[0], 0),
                lambda: zero_like_heatmap,
                lambda: tf.expand_dims(tf.sparse.to_dense(tf.sparse.SparseTensor(class_i_center, tf.ones_like(class_i_center[..., 0], tf.float32), dense_shape=[f_h, f_w]), validate_indices=False), axis=-1)
            )
            all_class_mask.append(class_i_mask)
        center_keypoint_heatmap = tf.concat(all_class_heatmap, axis=-1)
        center_keypoint_mask = tf.concat(all_class_mask, axis=-1)

        return center_keypoint_heatmap, center_keypoint_mask

    def _gen_center_offset_reg(self, ground_truth):
        pass

    def _gen_shape_reg(self, gound_truth):
        pass

def gaussian2D_tf_at_any_point(c_num, c_y, c_x, c_h, c_w, f_h, f_w):
    sigma = gaussian_radius_tf(c_h, c_w)
    c_y = tf.reshape(c_y, [-1, 1, 1])
    c_x = tf.reshape(c_x, [-1, 1, 1])

    y_range = tf.range(0, f_h, dtype=tf.float32)
    x_range = tf.range(0, f_w, dtype=tf.float32)
    [mesh_x, mesh_y] = tf.meshgrid(x_range, y_range)
    mesh_x = tf.expand_dims(mesh_x, 0)
    mesh_x = tf.tile(mesh_x, [c_num, 1, 1])
    mesh_y = tf.expand_dims(mesh_y, 0)
    mesh_y = tf.tile(mesh_y, [c_num, 1, 1])
    center_keypoint_heatmap = tf.exp(-((c_y-mesh_y)**2+(c_x-mesh_x)**2)/(2*sigma**2))
    return center_keypoint_heatmap

def gaussian2D(shape, sigma=1):
    # m, n = [(ss - 1.) / 2. for ss in shape]
    m, n = (shape[0]-1.)/2, (shape[1]-1.)/2
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D_tf(shape, sigma=1):
    m, n = shape[0], shape[1]
    m = tf.cast((m-1.)/2, dtype=tf.float32)
    n = tf.cast((n-1.)/2, dtype=tf.float32)

    y = tf.range(-m, m+1, dtype=tf.float32)
    x = tf.range(-n, n+1, dtype=tf.float32)
    [n_x, m_y] = tf.meshgrid(x, y)

    h = tf.exp(-(x * x + y * y) / (2 * sigma * sigma))
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

def gaussian_radius(height, width, min_overlap=0.7):
    """
    Args:
        height, width: Both are the array of the same shape (N,)!
    """

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return np.min([r1, r2, r3])

def draw_gaussian(heatmap, layer, center, sigma):
    # 目标热图只采用 Gaussian类型
    (HEAT_MAP_HEIGHT, HEAT_MAP_WIDTH) = heatmap.shape[0:2]
    temperature_size = sigma * 3
    TOP, BOTTOM = LEFT, RIGHT = Y_, X_ = 0, 1  # 纯粹为了可读
    mu_y = int(center[0]+0.5)  # 调整到HeatMap的坐标系，四舍五入.
    mu_x = int(center[1]+0.5)
    # 检查Gaussian_bounds是否落在HEATMAP之外,直接跳出运行,不支持不可见JOINT POINT
    left_top = [int(mu_y - temperature_size), int(mu_x - temperature_size)]
    right_bottom = [int(mu_y + temperature_size), int(mu_x + temperature_size)]
    if left_top[Y_] >= HEAT_MAP_HEIGHT or left_top[X_] >= HEAT_MAP_WIDTH or right_bottom[Y_] < 0 or right_bottom[X_] < 0:
        assert False

    # 生成Gaussian_Area
    size = temperature_size*2+1
    g = gaussian2D([size, size], sigma)
    # 确定可用Gaussian_Area
    g_y = max(0, -left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_HEIGHT) - left_top[Y_]
    g_x = max(0, -left_top[X_]), min(right_bottom[X_], HEAT_MAP_WIDTH) - left_top[X_]
    #
    heatmap_y = max(0, left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_HEIGHT)
    heatmap_x = max(0, left_top[X_]), min(right_bottom[X_], HEAT_MAP_WIDTH)

    heatmap[heatmap_y[TOP]:heatmap_y[BOTTOM], heatmap_x[LEFT]:heatmap_x[RIGHT], layer] = g[g_y[TOP]:g_y[BOTTOM], g_x[LEFT]:g_x[RIGHT]]
    pass

class ImageNormalizer:
    """
    每一类数据应当有不同的，应当自行去统计自己的训练数据!
    但这里暂时还是用voc的数据集里的东西！
    """

    def __init__(self):
        self._offset = (0.485, 0.456, 0.406)
        self._scale = (0.229, 0.224, 0.225)

    def __call__(self, image):
        """Normalizes the image to zero mean and unit variance."""
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        offset = tf.constant(self._offset)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset

        scale = tf.constant(self._scale)
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        return image
