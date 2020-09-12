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

class Scaler(KL.Layer):
    """特殊的scale层
    """

    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(Scaler, self).build(input_shape)
        self.built = True

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * KB.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * KB.sigmoid(-self.scale)
        return inputs * KB.sqrt(scale)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def enc_layer_begin_block(filters, is_sn=False):
    """
    enc_layer_begin_block和enc_layer_block的区别就在与，非SN情况下，少个BN层。
    """
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
        net = KL.ReLU(name='dec_relu_{}'.format(layer+1))(net)
        return net
    return forward

def dec_layer_extra_block(filters, layer):
    def forward(net):
        net = KL.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False, name='dec_extra_conv2d_{}'.format(layer))(net)
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
    """

    def __init__(self,
                 depth=64,       # 中间层通道数的计数单位
                 z_dim=100,      # 隐含向量的长度
                 image_size=64,  # 事实上这定义了输入、生成图像的规格！
                 num_outputs=3):
        """Constructor.
        Args:
            depth: Number of channels in last deconvolution layer(or first convolution layer) of the decoder(or encoder) network.
            z_dim:
            num_outputs: Nuber of output features. For images, this is the number of channels.
            image_size: the size of the image.
        """
        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._image_log2_size = int(math.log(image_size, 2))
        self._num_layers = int(math.log(image_size, 2))-3
        self._num_outputs = num_outputs

    def f_forward(self, input, name='feat', is_sn=False):
        """F network.The common structure of the Encoder and Discriminator!
        Args:
            input: with shape (self._image_size, self._image_size, self._num_outputs)
            name: name for the features
            is_sn: 是否spectral normalization。我只在Discriminator中用。
        Returns:
            features: with shape (BN, 4, 4, self._depth*2**( self._image_log2_size-3))
        """
        nl = self._num_layers
        nd = self._depth

        net = input
        # 3->self._depth
        net = enc_layer_begin_block(filters=nd, is_sn=is_sn)(net)
        # self._depth -> self._depth*2**nl
        for i in range(EXTRA_LAYERS_NUM):
            net = enc_layer_extra_block(filters=nd, layer=i, is_sn=is_sn)(net)
        for i in range(nl//2):
            nd = nd*2
            net = enc_layer_block(filters=nd, layer=i, is_sn=is_sn)(net)
        net = Attention(nd, name='enc_attention')(net)
        for i in range(nl//2, nl):
            nd = nd * 2
            net = enc_layer_block(filters=nd, layer=i, is_sn=is_sn)(net)

        # 此时nd == self._depth*2**nl == self._depth*2**(self._image_log2_size-3)
        # 此时: net 具有维度形状 BNx4x4x(depth*2**( self._image_log2_size-3))  #注意，这和标准的DCGAN还是有些区别的！
        # 注意: 4 = 2**(self._image_log2_size-1-(self._image_log2_size-3)).
        # 解释: "-1", 在enc_layer_begin_block里完成；"-(self._image_log2_size-3)", 在那些enc_layer_block里完成.
        # 赋名以检查.
        features = KL.Reshape(target_shape=(4, 4, nd), name='{}'.format(name))(net)
        return features

    def f_to_c(self, feat, name="critics", is_sn=False):
        """F->C ==> Discriminator
        Args:
            in_feature: with shape (BN, 4, 4, self._depth*2**( self._image_log2_size-3))
            name: name for the critics
        Returns:
            critics: with shape (BN, 1). So called, as it is not activated by sigmoid.
        """
        # 这个要严格和F的输出对上
        net = feat
        if is_sn:
            net = SN(KL.Conv2D(1, (4, 4), strides=1, padding='VALID', use_bias=False, name='sn_{}_raw'.format(name)))(net)
        else:
            net = KL.Conv2D(1, (4, 4), strides=1, padding='VALID', use_bias=False, name='{}_raw'.format(name))(net)
        # 此时：BNx1x1x1
        c = KL.Reshape((1,), name='{}'.format(name))(net)
        # 事实上，此处我总是有些犯糊涂，最后一层是没必要sigmoid激活的。
        # 即使在DCGAN的情况下，我们用的bce，他自己就加了sigmoid！！！

        return c

    def f_to_z(self, feat, name="z", is_sn=False):
        """F->Z ==> Encoder
        Args:
            in_feature: with shape (BN, 4, 4, self._depth*2**( self._image_log2_size-3))
            name: name for the z
        returns:
            z : with shape (BN, self._z_dim)
        """
        nz = self._z_dim
        net = feat

        if is_sn:
            net = SN(KL.Conv2D(nz, (4, 4), strides=1, padding='VALID', use_bias=False))(net)
        else:
            net = KL.Conv2D(nz, (4, 4), strides=1, padding='VALID', use_bias=False)(net)
        z = KL.Reshape((nz,), name='{}'.format(name))(net)
        return z

    def f_to_vaez(self, feat, name="vae_z"):
        nz = self._z_dim
        net = feat

        scaler = Scaler()
        z_mean = KL.Conv2D(nz, (4, 4), strides=1, padding='VALID', use_bias=False)(net)  # 此时：BNx1x1xnz
        z_mean = KL.BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_mean)  # 看文章“A Batch Normalized Inference Network Keeps the KL Vanishing Away”
        z_mean = KL.Reshape((nz,))(z_mean)  # 此时：BNxnz
        z_mean = scaler(z_mean, mode='positive')

        z_log_var = KL.Conv2D(nz, (4, 4), strides=1, padding='VALID', use_bias=False)(net)  # 此时：BNx1x1xnz
        z_log_var = KL.BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_log_var)   # 看文章“A Batch Normalized Inference Network Keeps the KL Vanishing Away”
        z_log_var = KL.Reshape((nz,))(z_log_var)  # 此时：BNxnz
        z_log_var = scaler(z_log_var, mode='negative')

        epsilon = KB.random_normal(shape=KB.shape(z_mean))                                                                 # 此时：BNxnz
        vae_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # 赋名以检查
        vae_z = KL.Reshape((nz,), name='{}'.format(name))(vae_z)
        return vae_z, kl_loss

    def z_to_c(self, z, name="critics", is_sn=False):
        """
        F->Z-C ==> D=T(E) : 有时可考虑D是否可做为E呢？
        """
        nz = self._z_dim
        net = KL.Reshape((1, 1, nz))(z)
        if is_sn:
            net = SN(KL.Conv2D(1, (1, 1), strides=1, padding='VALID', use_bias=False))(net)
        else:
            net = KL.Conv2D(1, (1, 1), strides=1, padding='VALID', use_bias=False)(net)
        c = KL.Reshape((1,), name='{}'.format(name))(net)
        return c

    def g_forward(self, z, name='image'):
        """Construct generator network from inputs to the final endpoint.
        Args:
            z: with shape (BN, self._z_dim)
        Returns:
            image: 生成的结果. with shape (BN, self._image_size, self._image_size, self._num_outputs)
        """
        nl = self._num_layers
        nd = self._depth * 2**nl
        nz = self._z_dim

        #
        net = KL.Reshape(target_shape=(1, 1, nz))(z)
        # nz -> nd
        net = dec_layer_init_block(filters=nd)(net)
        # nd -> self._depth
        for i in range(nl//2):
            nd = nd // 2
            net = dec_layer_block(filters=nd, layer=i)(net)
        # 中途加入一个Attention层!
        net = Attention(nd, name='dec_attention')(net)
        for i in range(nl//2, nl):
            nd = nd // 2
            net = dec_layer_block(filters=nd, layer=i)(net)
        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = dec_layer_extra_block(filters=nd, layer=i)(net)
        # self._depth -> 3
        net = dec_layer_end_block(self._num_outputs)(net)
        # 赋名以检查.
        image = KL.Reshape(target_shape=(self._image_size, self._image_size, self._num_outputs), name="{}".format(name))(net)

        return image

    def E(self, name='E', is_sn=False):
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        feat = self.f_forward(x, is_sn=is_sn)
        z = self.f_to_z(feat, is_sn=is_sn)

        model = KM.Model(inputs=x, outputs=[feat, z], name=name)
        return model

    def D(self, name='D', is_sn=True, is_te=False):
        """
        Args:
            name: Discrinator的名
            is_sn: 按SNGAN的原始论文的介绍，Discriminator里Batch Normalization和Spectral Normalization似乎不能共存!
                   默认情况，Discriminator均采用SN。
            is_te: Discriminator是否采用D=T(E)这种情况。
        """
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        feat = self.f_forward(x, is_sn=is_sn)
        if not is_te:
            critics = self.f_to_c(feat, is_sn=is_sn)
            model = KM.Model(inputs=x, outputs=[feat, critics], name=name)
        else:
            z = self.f_to_z(feat, is_sn=is_sn)
            critics = self.z_to_c(z, is_sn=is_sn)
            model = KM.Model(inputs=x, outputs=[feat, z, critics], name=name)
        return model

    def G(self, name='G'):
        """
        最终Generator里还是不用Spectral Normalization!
        """
        nz = self._z_dim
        z = KL.Input(shape=(nz,), name='z')
        image = self.g_forward(z)
        model = KM.Model(inputs=z, outputs=image, name=name)
        return model

    def AE(self, name='AE', is_vae=False):
        """
        有时E、G可以构成AutoEncoder情况。
        不考虑SN。
        """
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        feat = self.f_forward(x)
        if is_vae:
            z, kl_loss = self.f_to_vaez(feat)
        else:
            z = self.f_to_z(feat)
        image = self.g_forward(z)
        model = KM.Model(inputs=x, outputs=[z, image], name=name)
        if is_vae:
            model.add_loss(kl_loss)
        return model
