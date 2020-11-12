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
from tensorgroup.models.dataset.centernet_inputs import DefineInputs
from tensorgroup.models.dataset import mode_keys as ModeKey

"""
为提高效率，先将目录文件式的数据转换成tfrecord格式的数据。
"""


lanzhou_classes = {
    'single_white': 0,
    'double_red': 1,
    'word': 2
}
catenary_classes = {
    'holder': 0,
    'wire_holder': 1,
    'wire_hook': 2,
    'clamp_up': 3,
    'clamp_down': 4,
    'clamp_locator': 5,
    'insulator': 6,
    'support': 7
}

voc_custom_classes = {
    'lanzhou': lanzhou_classes,
    'catenary': catenary_classes
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


def xml_to_example(xmlpath, imgpath, classes):
    xml = etree.parse(xmlpath)
    root = xml.getroot()
    imgname = root.find('filename').text
    imgname = os.path.join(imgpath, imgname)
    image = tf.io.gfile.GFile(imgname, 'rb').read()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)      # labelImg 有bug，它保存的depth有问题，因此标记数据后，我有个修复操作。
    shape = np.asarray([height, width, depth], np.int32)
    xpath = xml.xpath('//object')
    ground_truth = np.zeros([len(xpath), 5], np.float32)
    for i in range(len(xpath)):
        obj = xpath[i]
        # print(obj.find('name').text)
        classid = classes[obj.find('name').text]
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
        'image': bytes_feature(image),   # 注意image是没有解析的。以后用tf.io.decode_jpeg来解析， 避免BGR、RGB搞不清的格式问题。
        'shape': bytes_feature(shape.tobytes()),
        'ground_truth': bytes_feature(ground_truth.tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(
        feature=features))
    return example


def dataset2tfrecord(xml_dir, img_dir, output_dir, name, datasetName='lanzhou', total_shards=2):
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    tf.io.gfile.mkdir(output_dir)
    outputfiles = []
    xmllist = tf.io.gfile.glob(os.path.join(xml_dir, '*.xml'))
    num_per_shard = int(math.ceil(len(xmllist)) / float(total_shards))
    object_classes = voc_custom_classes[datasetName]
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
                example = xml_to_example(xmllist[i], img_dir, object_classes)
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles


class VocCustomInput:
    """
    基本上将VocCustomInput和VocInput重构了差不多了，后面想把他们基类化，看时间吧！
    """

    def __init__(self,
                 tfrecords_dir,
                 datasetName='lanzhou',
                 inputs_definer=DefineInputs,
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
        self._num_classes = len(voc_custom_classes[datasetName])
        self._max_objects = max_objects
        self._inputs_definer = inputs_definer

    class Decoder:
        def __init__(self, image_normalizer, channels):
            self._image_normalizer = image_normalizer
            self._channels = channels

        def __call__(self, tfrecord):
            features = tf.io.parse_single_example(tfrecord, features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'shape': tf.io.FixedLenFeature([], tf.string),
                'ground_truth': tf.io.FixedLenFeature([], tf.string)
            })
            # shape = tf.io.decode_raw(features['shape'], tf.int32)
            ground_truth = tf.io.decode_raw(features['ground_truth'], tf.float32)
            # shape = tf.reshape(shape, [3])  # 可能原始XML中有关shape的信息不准，导致这儿的shape里的信息不可用。
            ground_truth = tf.reshape(ground_truth, [-1, 5])
            image = tf.io.decode_jpeg(features['image'])
            # image = tf.reshape(image, shape)
            if self._image_normalizer is not None:
                image = self._image_normalizer(image)
            return image, ground_truth

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

    def __call__(self, network_input_config):
        tfrecords = tf.io.gfile.glob(os.path.join(self._tfrecords_dir, TFR_PATTERN.format(self._mode)))
        dataset = tf.data.TFRecordDataset(tfrecords)
        return self.__gen_input__(dataset, network_input_config)

    def __gen_input__(self, dataset, network_input_config):
        decoder = self.Decoder(self.ImageNormalizer(), network_input_config['network_input_channels'])
        inputs_def = self._inputs_definer(network_input_config,
                                          num_classes=self._num_classes,
                                          max_objects=self._max_objects)
        dataset = dataset.map(decoder).map(inputs_def)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)
        if self._repeat_num > 0:
            dataset = dataset.repeat(self._repeat_num)
        if self._buffer_size > 0:
            dataset = dataset.shuffle(self._buffer_size)
        if self._batch_size > 0:
            dataset = dataset.batch(self._batch_size, drop_remainder=True)

        return dataset
