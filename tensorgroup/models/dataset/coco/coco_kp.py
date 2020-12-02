
import os
import io
import re
import threading

import numpy as np
import tensorflow as tf
import json
from PIL import Image
import base64

from typing import Text, Optional
from tensorgroup.models.dataset.keypointnet_inputs import DefineInputs
from tensorgroup.models.dataset import mode_keys as ModeKey

POINT_RADIUS = 10  # 5的来源：点当作21*21的对象。这样可使得尽量重用centernet中的代码.
TFR_PATTERN = '{}*.tfrecord'
# dataset_utils

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


PC_NUMBERS = 2
PC = {
    'point_1': 0,
    'point_2': 1,
    'p_c': 0,
    'p_b': 1,
    'p_r': 1
}

def get_annotations_dict(annotation_json_path):
    """
    从LabelMe文件中读取相关数据.了解LabelMe的文件格式是必要的.
    """
    if not os.path.exists(annotation_json_path):
        return None
    # 读入json文件
    with open(annotation_json_path, 'r') as f:
        json_text = json.load(f)
    #
    shapes = json_text.get('shapes', None)
    if shapes is None:
        return None
    im_w = json_text.get('imageWidth')
    im_h = json_text.get('imageHeight')

    base64_data = json_text.get('imageData')
    encoded_jpg = base64.b64decode(base64_data)
    #
    points = []
    for mark in shapes:
        shape_type = mark.get('shape_type')
        if not (shape_type == 'point'):
            continue

        pc = PC[mark.get('label')]
        [c_x, c_y] = np.array(mark.get('points'), dtype=np.int)[0]
        ymin, ymax, xmin, xmax = (c_y - POINT_RADIUS)/im_h, (c_y + POINT_RADIUS)/im_h, (c_x - POINT_RADIUS)/im_w, (c_x + POINT_RADIUS)/im_w
        points.extend([ymin, ymax, xmin, xmax, pc])  # 一维N*5个元素.

    annotation_dict = {'encoded_jpg': encoded_jpg,
                       'points': np.array(points, dtype=np.float32)}
    return annotation_dict


def create_tf_example(annotation_dict):

    features = {'image': bytes_feature(annotation_dict['encoded_jpg']),
                'ground_truth': bytes_feature(annotation_dict['points'].tobytes())
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def find_json_files(data_dir):
    pattern = os.path.join(data_dir, '*.json')
    filenames = tf.io.gfile.glob(pattern)
    return filenames


def produce_dataset_from_jsons(dataset_name, json_source_dir, target_directory):
    filenames = find_json_files(data_dir=json_source_dir)
    output_filename = '%s.tfrecord' % (dataset_name)
    output_file = os.path.join(target_directory, output_filename)
    writer = tf.io.TFRecordWriter(output_file)
    for filename in filenames:
        ann = get_annotations_dict(filename)
        example = create_tf_example(ann)
        writer.write(example.SerializeToString())
    writer.close()


class CocoKpInput:
    def __init__(self,
                 tfrecords_dir,
                 datasetName='three_point',
                 inputs_definer=DefineInputs,
                 mode: Text = ModeKey.TRAIN,
                 batch_size: Optional[int] = -1,
                 num_exsamples: Optional[int] = -1,
                 repeat_num: Optional[int] = -1,
                 buffer_size: Optional[int] = -1,
                 max_objects: Optional[int] = 10):
        assert mode is not None
        self._tfrecords_dir = tfrecords_dir
        self._mode = mode
        self._batch_size = batch_size
        self._num_examples = num_exsamples
        self._repeat_num = repeat_num
        self._buffer_size = buffer_size
        #
        self._num_classes = PC_NUMBERS        # 点的类型数，这个地方需要可配置。
        self._max_objects = max_objects  # 允许的点对象最大个数。
        self._inputs_definer = inputs_definer

    class Decoder:
        def __init__(self, image_normalizer, channels):
            self._image_normalizer = image_normalizer
            self._channels = channels

        def __call__(self, tfrecord):
            features = tf.io.parse_single_example(tfrecord, features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'ground_truth': tf.io.FixedLenFeature([], tf.string)
            })
            # shape = tf.io.decode_raw(features['shape'], tf.int32)
            ground_truth = tf.io.decode_raw(features['ground_truth'], tf.float32)
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
            # offset = tf.constant(self._offset)
            # offset = tf.expand_dims(offset, axis=0)
            # offset = tf.expand_dims(offset, axis=0)
            # image -= offset

            # scale = tf.constant(self._scale)
            # scale = tf.expand_dims(scale, axis=0)
            # scale = tf.expand_dims(scale, axis=0)
            # image /= scale
            return image

    def __call__(self, network_input_config):
        tfrecords = tf.io.gfile.glob(os.path.join(self._tfrecords_dir, TFR_PATTERN.format(self._mode)))
        print(tfrecords)
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
