import tensorflow as tf
import numpy as np
import os

import os
import numpy as np
import math
import sys

from typing import Text, Optional
from tensorgroup.models.dataset.u_net_inputs import DefineInputs
from tensorgroup.models.dataset import mode_keys as ModeKey

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

def img_to_example(image_name, img_dir, mask_ann_dir):
    mask_name = os.path.join(mask_ann_dir, image_name + ".png")
    image_name = os.path.join(img_dir, image_name)
    mask = tf.io.gfile.GFile(mask_name, 'rb').read()
    image = tf.io.gfile.GFile(image_name, 'rb').read()
    features = {
        'image': bytes_feature(image),
        'mask': bytes_feature(mask)
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def dataset2tfrecord(img_dir, mask_ann_dir, output_dir, name, total_shards=2):
    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)
    tf.io.gfile.mkdir(output_dir)
    outputfiles = []
    img_list = tf.io.gfile.glob(os.path.join(img_dir, '*.jpg'))
    num_per_shard = int(math.ceil(len(img_list)) / float(total_shards))
    for shard_id in range(total_shards):
        outputname = '%s_%05d-of-%05d.tfrecords' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)
        with tf.io.TFRecordWriter(outputname) as tf_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(img_list))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % (
                    i+1, len(img_list), shard_id+1, total_shards))
                sys.stdout.flush()
                print(img_list[i])
                img_name = os.path.basename(img_list[i])
                example = img_to_example(img_name, img_dir, mask_ann_dir)
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles

class MaskInputs:
    def __init__(self,
                 tfrecords_dir,
                 inputs_definer=DefineInputs,
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
        self._inputs_definer = inputs_definer

    class Decoder:
        def __init__(self, image_normalizer):
            self._image_normalizer = image_normalizer

        def __call__(self, tfrecord):
            features = tf.io.parse_single_example(tfrecord, features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'mask': tf.io.FixedLenFeature([], tf.string)
            })
            image = tf.image.decode_jpeg(features['image'], channels=3)
            mask = tf.image.decode_png(features['mask'], channels=1)
            if self._image_normalizer is not None:
                image, mask = self._image_normalizer(image, mask)
            return image, mask

    class ImageNormalizer:
        """
        每一类数据应当有不同的，应当自行去统计自己的训练数据!
        但这里暂时还是用voc的数据集里的东西！
        """

        def __init__(self):
            self._offset = (0.485, 0.456, 0.406)
            self._scale = (0.229, 0.224, 0.225)

        def __call__(self, image, mask):
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

            mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)
            return image, mask


    def __call__(self, network_input_config):
        tfrecords = tf.io.gfile.glob(os.path.join(self._tfrecords_dir, TFR_PATTERN.format(self._mode)))
        dataset = tf.data.TFRecordDataset(tfrecords)
        return self.__gen_input__(dataset, network_input_config)

    def __gen_input__(self, dataset, network_input_config):
        decoder = self.Decoder(self.ImageNormalizer())
        inputs_def = self._inputs_definer(network_input_config['network_input_shape'])
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
