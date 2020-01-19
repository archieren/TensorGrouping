import tensorflow as tf
import tensorflow_datasets as tfds

from tensorgroup.models.dataset.image_augmentor import image_augmentor


def reform_voc_for_train(config):
    def reformer(features):
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
        objects = features['objects']
        bbox = objects['bbox']
        ymin = tf.reshape(bbox[:,0], [-1, 1])
        xmin = tf.reshape(bbox[:,1], [-1, 1])
        ymax = tf.reshape(bbox[:,2], [-1, 1])
        xmax = tf.reshape(bbox[:,3], [-1, 1])
        classid = tf.dtypes.cast(objects['label'], dtype = tf.float32)
        classid = tf.reshape(classid, [-1, 1])
        ground_truth = tf.concat([ymin, ymax, xmin, xmax, classid], axis=1) # yyxx形式！
        # :param ground_truth: [ymin, ymax, xmin, xmax, classid]
        image = normalize_image(image)
        image, ground_truth = image_augmentor(  image=image,
                                                ground_truth=ground_truth,
                                                **config
                                                )
        return image , ground_truth
    return reformer


def normalize_image(image,
                    offset  =(0.485, 0.456, 0.406),
                    scale   =(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def get_voc_generator():
    """
    得到voc/2012相关数据集！
    """
    """

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
    _, info_voc = tfds.load( name="voc/2012"
                            , split="train"
                            #, with_info=True
                            #, decoders={'image': tfds.decode.SkipDecoding(),}
                            )
    print(info_voc)    