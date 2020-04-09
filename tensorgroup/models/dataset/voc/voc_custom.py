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
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ground_truth[i, :] = np.asarray([ymin/height, ymax/height, xmin/width, xmax/width, classid],
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


def parse_function(data, config):
    features = tf.parse_single_example(data, features={
        'image': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string),
        'ground_truth': tf.FixedLenFeature([], tf.string)
    })
    shape = tf.decode_raw(features['shape'], tf.int32)
    ground_truth = tf.decode_raw(features['ground_truth'], tf.float32)
    shape = tf.reshape(shape, [3])
    ground_truth = tf.reshape(ground_truth, [-1, 5])
    images = tf.image.decode_jpeg(features['image'], channels=3)
    images = tf.reshape(images, shape)
    images, ground_truth = image_augmentor(image=images,
                                           input_shape=shape,
                                           ground_truth=ground_truth,
                                           **config
                                           )
    return images, ground_truth


def get_generator(tfrecords, batch_size, buffer_size, image_preprocess_config):
    data = tf.data.TFRecordDataset(tfrecords)
    data = data.map(lambda x: parse_function(x, image_preprocess_config)).shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat()
    iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    init_op = iterator.make_initializer(data)
    return init_op, iterator

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
                 buffer_size: Optional[int] = -1):
        assert mode is not None
        self._tfrecords_dir = tfrecords_dir
        self._mode = mode
        self._batch_size = batch_size
        self._num_examples = num_exsamples
        self._repeat_num = repeat_num
        self._buffer_size = buffer_size
        #
        self._num_classes = len(voc_custom_classes)

    def __call__(self, image_augmentor_config):
        tfrecords = tf.io.gfile.glob(os.path.join(self._tfrecords_dir, TFR_PATTERN.format(self._mode)))
        print(tfrecords)
        dataset = tf.data.TFRecordDataset(tfrecords)
        inputs_def = DefineInputs(image_augmentor_config,
                                  num_classes=self._num_classes,
                                  image_normalizer=ImageNormalizer())
        dataset = dataset.map(inputs_def)  # 定义输入的内容、格式！ dataset = (image, heatmap)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)
        if self._repeat_num > 0:
            dataset = dataset.repeat(self._repeat_num)
        if self._buffer_size > 0:
            dataset = dataset.shuffle(self._buffer_size)
        if self._batch_size > 0:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)

        return dataset

class DefineInputs:
    """
    """

    def __init__(self, config, num_classes, image_normalizer):
        """ 指定网络需要的输入格式。
        Args:
            config:
              {'data_format': 'channels_last',
               'output_shape': [512, 512],                           # Must match the network's input_shape!
               'flip_prob': [0., 0.5],
               'fill_mode': 'BILINEAR',
               'color_jitter_prob': 0.5,
               'pad_truth_to': 100,                                   # Must match the maximal objects!
              }
            num_classes:
            image_normalizer:
        """
        self._config = config
        self._image_normalizer = image_normalizer
        self._num_classes = num_classes

    def __call__(self, tfrecord):
        """将features转换成模型需要的形式
        Args:
            tfrecord: one record .
        Returns:
            images:
            ground_truth:

        """
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
        image = self._image_normalizer(image)
        image, ground_truth = image_augmentor(image=image,
                                              ground_truth=ground_truth,
                                              **self._config
                                              )
        # ground_truth: [y_center, x_center, height, width, classid]

        # heatmap = self._def_inputs(image, ground_truth)
        # return image, heatmap
        return image, ground_truth

    def _def_inputs(self, image, ground_truth):
        h, w = self._config['output_shape']
        heatmap = tf.zeros((h/4, w/4, self._num_classes), dtype=tf.dtypes.float32)
        assert ground_truth.shape[0] != 0, "wrong"
        return heatmap


class ImageNormalizer:
    """
    """

    def __init__(self):
        """
        每一类数据应当有不同的，应当自行去统计自己的训练数据!
        但这里暂时还是用voc的数据集里的东西！
        """
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
