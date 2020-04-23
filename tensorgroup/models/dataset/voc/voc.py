"""Tools about VOC Dataset"""
from typing import Text, Optional
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorgroup.models.dataset import mode_keys as ModeKey
from tensorgroup.models.dataset.centernet_inputs import DefineInputs

voc_classes = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


class VocInput:
    """
    定义Voc类训练库的什么呢？
    """

    def __init__(self,
                 inputs_definer=DefineInputs,
                 mode: Text = ModeKey.TRAIN,     # 还未用！但有用的！
                 batch_size: Optional[int] = -1,
                 num_exsamples: Optional[int] = -1,
                 repeat_num: Optional[int] = -1,
                 buffer_size: Optional[int] = -1,
                 max_objects: Optional[int] = 100):
        assert mode is not None
        self._mode = mode
        self._batch_size = batch_size
        self._num_examples = num_exsamples
        self._repeat_num = repeat_num
        self._buffer_size = buffer_size
        #
        self._num_classes = len(voc_classes)
        self._max_objects = max_objects
        self._inputs_definer = inputs_definer

    def __call__(self, network_input_config):
        dataset = _get_voc_dataset(split=self._mode)
        return self.__gen_input__(dataset, network_input_config)

    def __gen_input__(self, dataset, network_input_config):
        decoder = self.Decoder(self.ImageNormalizer())
        # 定义所期望的输入格式！ dataset = (normailized_and_resized_image, ground_truth, center_round, center_offset, shape_offset, center_keypoint_heatmap, center_keypoint_mask)
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

    class Decoder:
        def __init__(self, image_normalizer):
            self._image_normalizer = image_normalizer

        def __call__(self, features):
            """将features转换成模型需要的形式
            Args:
                features: 输进来的features当有如下形式：
                    FeaturesDict({
                        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
                        'image/filename': Text(shape=(), dtype=tf.string),
                        'labels': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
                        'labels_no_difficult': Sequence(ClassLabel(shape=(),
                                                                   dtype=tf.int64,
                                                                   num_classes=20)),
                        'objects': Sequence({
                            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
                            'is_difficult': Tensor(shape=(), dtype=tf.bool),
                            'is_truncated': Tensor(shape=(), dtype=tf.bool),
                            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=20),
                            'pose': ClassLabel(shape=(), dtype=tf.int64, num_classes=5),
                        }),
                    })
                    注：tensorflow-datasets 工具似乎统一了bbox, [ymin, xmin, ymax, xmax]
                config:

            Returns:
                images:
                ground_truth:

            #mean = tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32)
            #std = tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32)
            """
            image = features['image']

            # yxyx--> yyxx
            ground_truth = _get_groundtruth_in_yyxx(features)
            # ground_truth: [ymin, ymax, xmin, xmax, classid]
            image = self._image_normalizer(image)
            if self._image_normalizer is not None:
                image = self._image_normalizer(image)
            return image, ground_truth

    class ImageNormalizer:
        """
        """

        def __init__(self, dataset_name='voc/2012'):
            if dataset_name != 'voc/2012':
                raise ValueError('TODO')
            self._dataset_name = dataset_name
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


def _get_voc_dataset(split='train'):
    """
    得到voc/2012相关数据集！
    FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': Text(shape=(), dtype=tf.string),
        'labels': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
        'labels_no_difficult': Sequence(ClassLabel(shape=(), dtype=tf.int64, num_classes=20)),
        'objects': Sequence({
                'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
                'is_difficult': Tensor(shape=(), dtype=tf.bool),
                'is_truncated': Tensor(shape=(), dtype=tf.bool),
                'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=20),
                'pose': ClassLabel(shape=(), dtype=tf.int64, num_classes=5),
        }),
    })

    Args:
         split:指定用那个库
    """
    dataset, _ = tfds.load(name="voc/2012", split=split, with_info=True
                           # , decoders={'image': tfds.decode.SkipDecoding(),}
                           )
    return dataset

def _get_groundtruth_in_yyxx(features):
    """
    VOC的数据集中，xy_坐标均归一化了的。
    """
    objects = features['objects']
    bbox = objects['bbox']      # yxyx形式
    ymin = tf.reshape(bbox[:, 0], [-1, 1])
    xmin = tf.reshape(bbox[:, 1], [-1, 1])
    ymax = tf.reshape(bbox[:, 2], [-1, 1])
    xmax = tf.reshape(bbox[:, 3], [-1, 1])
    classid = tf.dtypes.cast(objects['label'], dtype=tf.float32)
    classid = tf.reshape(classid, [-1, 1])
    ground_truth = tf.concat([ymin, ymax, xmin, xmax, classid], axis=1)  # yyxx形式！
    return ground_truth
