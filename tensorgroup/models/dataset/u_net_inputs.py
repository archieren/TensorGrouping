import tensorflow as tf
import numpy as np
import os


class DefineInputs:
    """
    """
    def __init__(self, i_size=[1024, 1024]):
        self.i_size = i_size
        pass

    def __call__(self, image, mask):
        image = tf.image.resize(image, self.i_size)
        mask = tf.image.resize(mask, self.i_size)
        mask = tf.concat([mask, mask, mask, mask, mask, mask, mask], axis=-1)
        return ({'image': image}, {'u_2__net': mask})


