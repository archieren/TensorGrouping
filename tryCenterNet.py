import os
import hashlib

import multiprocessing
import numpy as np
import skimage.io as io

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds


from tensorgroup.models.networks.ResnetKeypointBuilder import ResnetKeypointBuilder as RKB
from tensorgroup.models.networks.BagnetBuilder import BagnetBuilder as BB
from tensorgroup.models.networks.CenterNetBuilder import CenterNetBuilder as CNB 


KA = tf.keras.applications
KL = tf.keras.layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


image_shape = (384, 384, 3)

lr = 0.001
batch_size = 15
buffer_size = 256
epochs = 160
reduce_lr_epoch = []
config = {
    'mode': 'train',                                       # 'train', 'test'
    'input_size': 384,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.1,                                 
    'top_k_results_output': 100,                           

}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [384, 384],
    'zoom_size': [400, 400],
    'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    'color_jitter_prob': 0.5,
    'rotate': [0.5, -5., -5.],
    'pad_truth_to': 60,
}

def about_dataset_voc():
    from tensorgroup.models.dataset.voc import voc
    train_voc, _ = tfds.load( name="voc/2012"
                            , split="train"
                            , with_info=True
                            #, decoders={'image': tfds.decode.SkipDecoding(),}
                            )
    fun = voc.reform_voc_for_train(image_augmentor_config)
    for image, gt in train_voc.map(fun).take(2):
        plt.imshow(image)
        plt.show()
        print(gt)


if __name__ == '__main__':
    about_dataset_voc()