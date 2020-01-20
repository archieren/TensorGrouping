
from typing import Text, Optional
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorgroup.models.dataset.image_augmentor import image_augmentor
from tensorgroup.models.dataset import mode_keys as ModeKey

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

class VocInput(object):    #我在这里用Functional Object提供自己一个范式！
    def __init__(self
                , mode :Text = ModeKey.TRAIN     # 还未用！但有用的！
                , batch_size:int = 2
                , num_exsamples:int = 10):
        assert mode is not None
        self._mode = mode
        self._batch_size= batch_size
        self._num_examples= num_exsamples

    def __call__(self, config):
        if self._mode == ModeKey.TRAIN :
            dataset=self._get_voc_dataset(split='train')
            inputs_def = Define_Inputs(config,image_normalizer = Image_Normalizer())
            dataset = dataset.map(inputs_def) # 定义输入的内容、格式！
            dataset = dataset.take(self._num_examples)
            dataset = dataset.batch(self._batch_size)
            return dataset
        else :
            raise ValueError("TODO")

    def _get_voc_dataset(self,split = 'train'):
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
        """
        dataset, _ = tfds.load( name="voc/2012"
                                , split= split
                                , with_info=True
                                #, decoders={'image': tfds.decode.SkipDecoding(),}
                                )
        return dataset   

class Define_Inputs(object):
    def __init__(self, config,image_normalizer):
        self._config = config
        self._image_normalizer = image_normalizer


    def __call__(self,features):
        """将features转换成模型需要的形式
        Args:
            features: 输进来的features当有如下形式：
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
        ground_truth = self._get_groundtruth(features)
        #ground_truth: [ymin, ymax, xmin, xmax, classid]
        image = self._image_normalizer(image)
        image, ground_truth = image_augmentor(  image=image,
                                                ground_truth=ground_truth,
                                                **self._config
                                                )
        return image , ground_truth        

    def _get_groundtruth(self, features):
        objects = features['objects']
        bbox = objects['bbox']      # yxyx形式
        ymin = tf.reshape(bbox[:,0], [-1, 1])
        xmin = tf.reshape(bbox[:,1], [-1, 1])
        ymax = tf.reshape(bbox[:,2], [-1, 1])
        xmax = tf.reshape(bbox[:,3], [-1, 1])
        classid = tf.dtypes.cast(objects['label'], dtype = tf.float32)
        classid = tf.reshape(classid, [-1, 1])
        ground_truth = tf.concat([ymin, ymax, xmin, xmax, classid], axis=1) # yyxx形式！
        return ground_truth

    def def_inputs(self, image, ground_truth):
        pass

class Image_Normalizer:
    def __init__(self, dataset_name = 'voc/2017'):
        if not dataset_name == 'voc/2017':
            raise ValueError('TODO')
        self._dataset_name = dataset_name
        self._offset    = (0.485, 0.456, 0.406)
        self._scale     =(0.229, 0.224, 0.225)

    def __call__(self,image):
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


