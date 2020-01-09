
import multiprocessing

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

KA = tf.keras.applications
KL = tf.keras.layers


from  tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
from tensorgroup.models.networks.BagnetBuilder import BagnetBuilder as BB
from tensorgroup.models.networks.CenterNetBuilder import CenterNetBuilder as CNB 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


image_shape = (224, 224, 3)

def about_model_ResnetKeypoint():
    modelP = RKB.build_pose_resnet_50(input_shape = (224, 224, 3), num_outputs = 10)
    modelP.summary()

def about_model_BageNet():
    modelB = BB.build_bagnet_9() #input_shape = (224, 224, 3)
    modelB.compile(optimizer=tf.keras.optimizers.RMSprop(0.001)
                    ,loss='sparse_categorical_crossentropy'
                    ,metrics=['sparse_categorical_accuracy']
                    )
    modelB.summary()

def about_keras_model_ResNet50V2():
    modelD = KA.ResNet50V2( weights='imagenet'
                        ,input_tensor=KL.Input(shape=(32*7, 32*7, 3))   # 32*output_shape= input_shape
                        ,include_top=False
                        )
    modelD.summary()
    print(modelD.input)
    print(modelD.output)
    print(modelD.outputs)

def about_keras_model_InceptionResNetV2():
    modelC = KA.InceptionResNetV2( weights='imagenet'
                        ,input_tensor=KL.Input(shape=(299, 299, 3))  # 32*output_shape+43 = input_shape
                        ,include_top=False
                        )
    modelC.summary()

#print(tfds.list_builders())
#print( multiprocessing.cpu_count())

def about_keras_model_CenterNet():
    train_model, prediction_model, debug_model = CNB.CenterNetOnResNet50V2(1000)
    train_model.summary()
    prediction_model.summary()
    debug_model.summary()

def about_dataset_imagenet2012():

    """
    FeaturesDict({
        'file_name': Text(shape=(), dtype=tf.string),
        'image': Image(shape=(None, None, 3), dtype=tf.uint8),
        'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=1000),
    })
    """

    imagenet2012_train, info_train = tfds.load( name="imagenet2012"
                                                , split="train"
                                                , with_info=True
                                                #, decoders={'image': tfds.decode.SkipDecoding(),}
                                                )

    imagenet2012_val, info_val = tfds.load( name="imagenet2012"
                                            , split="validation"
                                            , with_info=True
                                            #, decoders={'image': tfds.decode.SkipDecoding(),}
                                            )
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

    _, info_voc= tfds.load( name="voc/2007"
                            #, split="train"
                            , with_info=True
                            #, decoders={'image': tfds.decode.SkipDecoding(),}
                            )
    print(info_voc)

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
                            #, split="train"
                            , with_info=True
                            #, decoders={'image': tfds.decode.SkipDecoding(),}
                            )
    print(info_voc)

def example():
    """
    def d_scale(record):
        image, label = record['image'], record['label']    
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label
    """

    imagenet2012_train, _ = tfds.load(  name="imagenet2012"
                                        , split="train"
                                        , with_info=True
                                        #, decoders={'image': tfds.decode.SkipDecoding(),}
                                        )

    #imagenet2012_train_pro = imagenet2012_train.map(d_scale)

    for example in imagenet2012_train.take(10):
        image = example['image']
        print(repr(example))
        plt.imshow(image)
        plt.show()

    def preprocess_image(record):
        image = record['image']
        label = record['label']
        image = tf.cast(image, tf.float32)
        image /= 255      
        image = tf.image.resize(image
                                , [224, 224]
                                , method=tf.image.ResizeMethod.BILINEAR
                                )
        return image, label

    imagenet2012_train = imagenet2012_train.map(preprocess_image)
    imagenet2012_train = imagenet2012_train.filter(lambda  image, label : label != 10).take(1024)
    print(imagenet2012_train)
    for images, _ in imagenet2012_train.batch(32):
        print(images.shape)

    modelB = BB.build_bagnet_9(input_shape = (224, 224, 3))
    modelB.compile(optimizer=tf.keras.optimizers.RMSprop(0.001)
                    ,loss='sparse_categorical_crossentropy'
                    ,metrics=['sparse_categorical_accuracy']
                    )    
    modelB.fit(imagenet2012_train.batch(8),epochs=1)


if __name__ == '__main__':
    #about_dataset_voc()
    #example()
    #about_keras_model_ResNet50V2()
    about_keras_model_CenterNet()