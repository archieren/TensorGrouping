
# import multiprocessing
import numpy as np
import skimage.io as io

from pycocotools.coco import COCO

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds


from tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
from tensorgroup.models.networks.BagnetBuilder import BagnetBuilder as BB
from tensorgroup.models.networks.CenterNetBuilder import CenterNetBuilder as CNB

import os
# import hashlib

# from tensorflow_datasets.core.download.download_manager_test import Artifact as AF
# from tensorflow_datasets.core.download import resource as RL

KA = tf.keras.applications
KL = tf.keras.layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


image_shape = (224, 224, 3)

def about_model_ResnetKeypoint():
    modelP = RKB.build_keypoint_resnet_50(input_shape=(224, 224, 3), num_outputs=10)
    modelP.summary()

def about_model_BageNet():
    modelB = BB.build_bagnet_9()  # input_shape = (224, 224, 3)
    modelB.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['sparse_categorical_accuracy'])
    modelB.summary()

def about_keras_model_ResNet50V2():
    modelD = KA.ResNet50V2(weights='imagenet',
                           input_tensor=KL.Input(shape=(32*7, 32*7, 3)),  # 32*output_shape= input_shape
                           include_top=False)
    modelD.summary()
    print(modelD.input)
    print(modelD.output)
    print(modelD.outputs)

def about_keras_model_InceptionResNetV2():
    modelC = KA.InceptionResNetV2(weights='imagenet',
                                  input_tensor=KL.Input(shape=(299, 299, 3)),  # 32*output_shape+43 = input_shape
                                  include_top=False)
    modelC.summary()


# print(tfds.list_builders())
# print( multiprocessing.cpu_count())

def about_keras_model_CenterNet(which="train"):
    train_model, prediction_model, debug_model = CNB.CenterNetOnResNet50V2(1000)
    if which == "train":
        train_model.summary()
    elif which == "pred":
        prediction_model.summary()
    elif which == "debug":
        debug_model.summary()
    else:
        print("About which model of the centernet")

def about_dataset_imagenet2012():
    """
    FeaturesDict({
        'file_name': Text(shape=(), dtype=tf.string),
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=1000),
    })
    """

    imagenet2012_train, info_train = tfds.load(name="imagenet2012",
                                               split="train",
                                               with_info=True)  # decoders={'image': tfds.decode.SkipDecoding(),}

    imagenet2012_val, info_val = tfds.load(name="imagenet2012",
                                           split="validation",
                                           with_info=True)  # decoders={'image': tfds.decode.SkipDecoding(),}
    assert isinstance(imagenet2012_train, tf.data.Dataset)
    assert isinstance(imagenet2012_val, tf.data.Dataset)
    print(info_train)
    print(info_val)

def about_dataset_voc():
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

    train_voc, info_voc = tfds.load(name="voc/2007",
                                    with_info=True)

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

    train_voc, info_voc = tfds.load(name="voc/2012", split="train", with_info=True)
    for example in train_voc.take(10):
        image = example['image']
        print(image.shape)
        plt.imshow(image)
        plt.show()


def about_dataset_coco():
    """
    FeaturesDict({
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'image/filename': Text(shape=(), dtype=tf.string),
        'image/id': Tensor(shape=(), dtype=tf.int64),
        'objects': Sequence({
            'area': Tensor(shape=(), dtype=tf.int64),
            'bbox': BBoxFeature(shape=(4,), dtype=tf.float32),
            'id': Tensor(shape=(), dtype=tf.int64),
            'is_crowd': Tensor(shape=(), dtype=tf.bool),
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=80),
        }),
    })
    """
    """
    x = RL.get_dl_fname("http://images.cocodataset.org/zips/test2017.zip"
                        ,"c7908c3c9f94ba2f3340ebbeec58c25db6be8774f18d68c2f15d0e369d95baba")
    print(x)

    x = RL.get_dl_fname("http://images.cocodataset.org/zips/train2017.zip"
                        ,"69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929")
    print(x)
    """

    _, info_coco = tfds.load(name="coco/2017", split="train", with_info=True)
    print(info_coco)


def about_dataset_fashion_mnist():
    dataset, metadata = tfds.load(name='fashion_mnist', as_supervised=True, with_info=True)
    # train_dataset, test_dataset = dataset['train'], dataset['test']
    print(metadata)


def about_coco_api():
    dataDir = os.path.expanduser('~/Data/COCO_2017')  # 要了解的是其目录结构！
    dataType = 'train2017'
    # annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    # 一切都从这儿开始
    # 它，提供各种各样的查询
    coco = COCO(annFile)
    """
    print(len(coco.getCatIds()))  #查询Cat的Ids
    print(len(coco.getImgIds()))  #查询图像的Ids
    cats = coco.loadCats(coco.getCatIds()) #
    print(cats[0])
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    """
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    # 加载并显示图片,可以使用两种方式: 1) 加载本地图片, 2) 在线加载远程图片
    # 1) 使用本地路径, 对应关键字 "file_name"
    image = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    # 2) 使用 url, 对应关键字 "coco_url"
    # I = io.imread(img['coco_url'])

    plt.axis('off')
    plt.imshow(image)
    plt.show()

    # 加载并显示标注信息
    plt.imshow(image)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()


def example():
    imagenet2012_train, _ = tfds.load(name="imagenet2012", split="train", with_info=True)

    # imagenet2012_train_pro = imagenet2012_train.map(d_scale)

    # for example in imagenet2012_train.take(10):
    #     image = example['image']
    #     print(repr(example))
    #     plt.imshow(image)
    #     plt.show()

    def preprocess_image(record):
        image = record['image']
        label = record['label']
        image = tf.cast(image, tf.float32)
        image /= 255
        image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
        return image, label

    imagenet2012_train = imagenet2012_train.map(preprocess_image)
    imagenet2012_train = imagenet2012_train.filter(lambda image, label: label != 10).take(8192)
    # print(imagenet2012_train)
    # for images, _ in imagenet2012_train.batch(32):
    #     print(images.shape)
    modelB = BB.build_bagnet_9(input_shape=(224, 224, 3))
    modelB.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['sparse_categorical_accuracy'])
    modelB.fit(imagenet2012_train.batch(8), epochs=1)


"""
TOSEE: How to make a gaussian kernel
import tensorflow as tf

# Make Gaussian kernel following SciPy logic
def make_gaussian_2d_kernel(sigma, truncate=4.0, dtype=tf.float32):
    radius = tf.to_int32(sigma * truncate)
    x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
    k = tf.exp(-0.5 * tf.square(x / sigma))
    k = k / tf.reduce_sum(k)
    return tf.expand_dims(k, 1) * k

# Input data
image = tf.placeholder(tf.float32, [16, 96, 96, 3])
# Convolution kernel
kernel = make_gaussian_2d_kernel(5)
# Apply kernel to each channel (see https://stackoverflow.com/q/55687616/1782792)
kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
image_filtered = tf.nn.separable_conv2d(
    image, kernel, tf.eye(3, batch_shape=[1, 1]),
    strides=[1, 1, 1, 1], padding='SAME')
"""
if __name__ == '__main__':
    # about_dataset_voc()
    example()
    # about_model_BageNet()
    # about_model_ResnetKeypoint()
    # about_keras_model_ResNet50V2()
    # about_keras_model_CenterNet("pred")
    # about_dataset_coco()
    # about_dataset_fashion_mnist()
    # about_coco_api()
