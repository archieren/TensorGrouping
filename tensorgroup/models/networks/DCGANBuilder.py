# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import tensorflow as tf
from tensorgroup.models.networks.layers.sa import Attention
from tensorgroup.models.networks.layers.sn import SpectralNormalization as SN

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

EXTRA_LAYERS_NUM = 0
# pylint: enable=unused-import

def enc_layer_begin_block(filters, is_sn=False):
    def forward(net):
        if not is_sn:
            net = KL.Conv2D(filters, (4, 4), strides=2, padding='same', name='enc_conv2d_{}'.format('init'), use_bias=False)(net)
        else:
            net = SN(KL.Conv2D(filters, (4, 4), strides=2, padding='same', use_bias=False), name='enc_sn_conv2d_{}'.format('init'))(net)
        net = KL.LeakyReLU(0.2, name='enc_leakyrelu_{}'.format('init'))(net)
        return net
    return forward

def enc_layer_extra_block(filters, layer, is_sn=False):
    def forward(net):
        if not is_sn:
            net = KL.Conv2D(filters, (3, 3), strides=1, padding='same', name='enc_extra_conv2d_{}'.format(layer), use_bias=False)(net)
            net = KL.BatchNormalization(name='enc_extra_batchnorm_{}'.format(layer))(net)
        else:
            net = SN(KL.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False), name='enc_extra_sn_conv2d_{}'.format(layer))(net)
        net = KL.LeakyReLU(0.2, name='enc_extra_leakyrelu_{}'.format(layer))(net)
        return net
    return forward

def enc_layer_block(filters, layer, is_sn=False):
    def forward(net):
        if not is_sn:
            net = KL.Conv2D(filters, (4, 4), strides=2, padding='same', name='enc_conv2d_{}'.format(layer+1), use_bias=False)(net)
            net = KL.BatchNormalization(name='enc_batchnorm_{}'.format(layer+1))(net)
        else:
            net = SN(KL.Conv2D(filters, (4, 4), strides=2, padding='same', use_bias=False), name='enc_sn_conv2d_{}'.format(layer+1))(net)
        net = KL.LeakyReLU(0.2, name='enc_leakyrelu_{}'.format(layer+1))(net)
        return net
    return forward

def dec_layer_init_block(filters):
    def forward(net):
        net = KL.Conv2DTranspose(filters, (4, 4), strides=1, padding='valid', use_bias=False, name='dec_init_conv2dtrans')(net)
        net = KL.BatchNormalization(name='dec_init_batchnorm')(net)
        net = KL.ReLU(name='dec_init_relu')(net)
        return net
    return forward

def dec_layer_block(filters, layer):
    def forward(net):
        net = KL.Conv2DTranspose(filters, (4, 4), strides=2, padding='same', use_bias=False, name='dec_conv2dtrans_{}'.format(layer+1))(net)
        net = KL.BatchNormalization(name='dec_batchnorm_{}'.format(layer+1))(net)
        net = KL.ReLU(name='dec_relu_{}'.format(layer))(net)
        return net
    return forward

def dec_layer_extra_block(filters, layer):
    def forward(net):
        net = KL.Conv2D(filters, (3, 3), strides=1, padding='same', name='dec_extra_conv2d_{}'.format(layer), use_bias=False)(net)
        net = KL.BatchNormalization(name='dec_extra_batchnorm_{}'.format(layer))(net)
        net = KL.ReLU(name='dec_extra_relu_{}'.format(layer))(net)
        return net
    return forward()

def dec_layer_end_block(filters):
    def forward(net):
        net = KL.Conv2DTranspose(filters, (4, 4), strides=2, padding='same', use_bias=False,
                                 name='dec_end_conv2dtrans')(net)
        net = KL.Activation('tanh', name='tanh_output')(net)
        return net
    return forward

class DCGANBuilder(object):
    """Implementation of DCGAN.
    F,Z for Encoder
    F, SN_F, Critic, SN_Critic for Discriminator
    """

    def __init__(self,
                 depth=64,
                 z_dim=100,      # 隐含向量的长度
                 image_size=64,  # 事实上这定义了输入、生成图像的规格！
                 num_outputs=3):
        """Constructor.
        Args:
            depth: Number of channels in last deconvolution layer(or first convolution layer) of
                the decoder(or encoder) network.
            final_size: The shape of the final output.
            num_outputs: Nuber of output features. For images, this is the
                number of channels.
            fused_batch_norm: If 'True', use a faster, fused implementation
                of batch normalization.
        """
        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._num_layers = int(math.log(image_size, 2))-3
        self._num_outputs = num_outputs

    def f_forward(self, input, is_sn=False):
        """F network.The common structure of the Encoder and Discriminator!
        Args:
            name:
        Returns:
            features:
            F->Z ==> E
            F->C ==> D
        """
        # net_input = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        # net = net_input
        net = input

        num_layers = self._num_layers
        current_depth = self._depth
        # 3->self._depth
        net = enc_layer_begin_block(filters=current_depth, is_sn=is_sn)(net)
        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = enc_layer_extra_block(filters=current_depth, layer=i, is_sn=is_sn)(net)

        ##
        for i in range(num_layers//2):
            current_depth = current_depth*2
            net = enc_layer_block(filters=current_depth, layer=i, is_sn=is_sn)(net)

        # 中途加入一个Attention层!
        net = Attention(current_depth, name='enc_attention')(net)
        # 中途加入一个Attention层!
        for i in range(num_layers//2, num_layers):
            current_depth = current_depth * 2
            net = enc_layer_block(filters=current_depth, layer=i, is_sn=is_sn)(net)

        # 此时current_depth == self._depth*2**(isize-3)
        # 令 isize = int(math.log(height, 2))
        # 此时: BNx4x4x(depth*2**( isize-3))  #注意，这和标准的DCGAN还是有些区别的！
        # model.summary()
        return net

    def f_to_c(self, in_feature, is_sn=False):
        # 这个要严格和F的输出对上
        net = in_feature
        if is_sn:
            net = SN(KL.Conv2D(1, (4, 4), strides=1, padding='VALID', use_bias=False, name='sn_critics_raw'))(net)
        else:
            net = KL.Conv2D(1, (4, 4), strides=1, padding='VALID', use_bias=False, name='critics_raw')(net)
        # 此时：BNx1x1x1
        net = KL.Reshape((1,), name='critics')(net)
        # 事实上，此处我总是有些犯糊涂，最后一层是没必要sigmoid激活的。
        # 即使在DCGAN的情况下，我们用的bce，他自己就加了sigmoid！！！

        return net

    def f_to_z(self, in_feature):
        net = in_feature
        net = KL.Conv2D(self._z_dim, (4, 4), strides=1, padding='VALID', use_bias=False)(net)
        net = KL.Reshape((self._z_dim,), name='z')(net)
        return net

    def g_forward(self, z):
        """源自 Generator network for DCGAN.
        Construct generator network from inputs to the final endpoint.
        Args:
        """
        num_layers = self._num_layers
        current_depth = self._depth * 2**num_layers

        #
        net = KL.Reshape((1, 1, self._z_dim))(z)

        net = dec_layer_init_block(filters=current_depth)(net)
        for i in range(num_layers//2):
            current_depth = current_depth // 2
            net = dec_layer_block(filters=current_depth, layer=i)(net)
        # 中途加入一个Attention层!
        net = Attention(current_depth, name='dec_attention')(net)
        for i in range(num_layers//2, num_layers):
            current_depth = current_depth // 2
            net = dec_layer_block(filters=current_depth, layer=i)(net)
        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = dec_layer_extra_block(filters=current_depth, layer=i)(net)
        # self._depth -> 3
        net = dec_layer_end_block(self._num_outputs)(net)

        return net

    def E(self, name='E', is_sn=False):
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        feat = self.f_forward(x, is_sn=is_sn)
        z = self.f_to_z(feat)

        model = KM.Model(inputs=x, outputs=[feat, z], name=name)
        return model

    def D(self, name='D', is_sn=True):
        """
        按SNGAN的原始论文的介绍，Discriminator里Batch Normalization和Spectral Normalization似乎不能共存!
        """
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        feat = self.f_forward(x, is_sn=is_sn)
        critics = self.f_to_c(feat, is_sn=is_sn)

        model = KM.Model(inputs=x, outputs=[feat, critics], name=name)
        # model.summary()
        return model

    def G(self, name='G'):
        """
        最终Generator里还是不用Spectral Normalization!
        """
        z = KL.Input(shape=(self._z_dim,), name='input')
        image = self.g_forward(z)
        model = KM.Model(inputs=z, outputs=image, name=name)
        return model
