import tensorflow as tf
import numpy as np
import os


class DefineInputs:
    """
    """
    def __init__(self, network_input_config, i_size=[1024, 1024], in_name='image', out_name='side_all'):
        self.i_size = network_input_config['network_input_size']
        self.in_name = network_input_config['in_name']
        self.out_name = network_input_config['out_name']
        pass

    def __call__(self, image, mask):
        image = tf.image.resize(image, self.i_size)
        mask = tf.image.resize(mask, self.i_size)
        if self.out_name == 'side_all':
            mask = tf.concat([mask, mask, mask, mask, mask, mask, mask], axis=-1)
        return ({self.in_name: image}, {self.out_name: mask})
