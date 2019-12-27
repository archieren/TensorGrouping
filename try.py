
import multiprocessing

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds


from  tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
from tensorgroup.models.networks.BagnetBuilder import BagnetBuilder as BB

#image_shape = (33, 33, 3)
#modelP = RKB.build_pose_resnet_50(input_shape = image_shape, num_outputs = 10)
#modelP.summary()
#modelB = BB.build_bagnet_33(input_shape=image_shape, num_outputs=1000)
#modelB.summary()

"""
FeaturesDict({
    'file_name': Text(shape=(), dtype=tf.string),
    'image': Image(shape=(None, None, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=1000),
})
"""

print(tfds.list_builders())

mnist_train, info_mnist_train= tfds.load(name="mnist", split="train", with_info= True)
print(info_mnist_train)
assert isinstance(mnist_train, tf.data.Dataset)

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
print(info_train)
print(info_val)

print( multiprocessing.cpu_count())

def d_scale(record):
    image, label = record['image'], record['label']    
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

imagenet2012_train_pro = imagenet2012_train.map(d_scale)
print(imagenet2012_train_pro.element_spec)
#print(imagenet2012_train.output_types)
#print(imagenet2012_train.output_shape)
#for example in imagenet2012_train.take(10):
    #image = example['image']
    #print(repr(example))
    #plt.imshow(image)
    #plt.show()


