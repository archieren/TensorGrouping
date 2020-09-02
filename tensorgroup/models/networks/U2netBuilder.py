# See "U-2-Net:Going Deeper with Nested U-Structure for Salient Object Detection".
import tensorflow as tf  # TF 2.0
import tensorgroup.models.networks.layers.rsu as RSU

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

def down_rsu_with(rsu, down_in, pooling=True):
    hor_out = rsu(down_in)
    if pooling:
        down_out = KL.MaxPool2D(pool_size=2, strides=2)(hor_out)
    else:
        down_out = hor_out
    return hor_out, down_out

def up_rsu_with(rsu, hor_in, up_in, up_down_scale, side_c, upsampling=True):
    if upsampling:
        up_out = KL.UpSampling2D(size=2, interpolation='bilinear')(up_in)
    else:
        up_out = up_in
    up_out = KL.concatenate([hor_in, up_out])
    up_out = rsu(up_out)
    side_fuse = KL.Conv2D(side_c, kernel_size=3, padding='same')(up_out)
    if up_down_scale > 0:
        side_fuse = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')(side_fuse)
    return side_fuse, up_out

def floor_rsu(rsu, down_in, up_down_scale, side_c):
    up_out = rsu(down_in)
    side_fuse = KL.Conv2D(side_c, kernel_size=3, padding='same')(up_out)
    if up_down_scale > 0:
        side_fuse = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')(side_fuse)
    return side_fuse, up_out

class U_2_Net(KM.Model):
    def __init__(self, side_c=1, is_simple=False, **kwargs):
        super(U_2_Net, self).__init__(**kwargs)
        self.side_c = side_c
        if not is_simple:
            self.rsu_d0 = RSU.RSU7(filters=64, mid_filters=32, name='rsu7_d')
            self.rsu_d1 = RSU.RSU6(filters=128, mid_filters=32, name='rsu6_d')
            self.rsu_d2 = RSU.RSU5(filters=256, mid_filters=64, name='rsu5_d')
            self.rsu_d3 = RSU.RSU4(filters=512, mid_filters=128, name='rsu4_d')
            self.rsu_d4 = RSU.RSU4F(filters=512, mid_filters=256, name='rsu4f_d')

            self.rsu_f = RSU.RSU4F(filters=512, mid_filters=256, name='rsu4f_f')

            self.rsu_u4 = RSU.RSU4F(filters=512, mid_filters=256, name='rsu4f_u')
            self.rsu_u3 = RSU.RSU4(filters=256, mid_filters=128, name='rsu4_u')
            self.rsu_u2 = RSU.RSU5(filters=128, mid_filters=64, name='rsu5_u')
            self.rsu_u1 = RSU.RSU6(filters=64, mid_filters=32, name='rsu6_u')
            self.rsu_u0 = RSU.RSU7(filters=64, mid_filters=16, name='rsu7_u')
        else:
            self.rsu_d0 = RSU.RSU7(filters=64, mid_filters=16, name='rsu7_d')
            self.rsu_d1 = RSU.RSU6(filters=64, mid_filters=16, name='rsu6_d')
            self.rsu_d2 = RSU.RSU5(filters=64, mid_filters=16, name='rsu5_d')
            self.rsu_d3 = RSU.RSU4(filters=64, mid_filters=16, name='rsu4_d')
            self.rsu_d4 = RSU.RSU4F(filters=64, mid_filters=16, name='rsu4f_d')

            self.rsu_f = RSU.RSU4F(filters=64, mid_filters=16, name='rsu4f_f')

            self.rsu_u4 = RSU.RSU4F(filters=64, mid_filters=16, name='rsu4f_u')
            self.rsu_u3 = RSU.RSU4(filters=64, mid_filters=16, name='rsu4_u')
            self.rsu_u2 = RSU.RSU5(filters=64, mid_filters=16, name='rsu5_u')
            self.rsu_u1 = RSU.RSU6(filters=64, mid_filters=16, name='rsu6_u')
            self.rsu_u0 = RSU.RSU7(filters=64, mid_filters=16, name='rsu7_u')

    def call(self, inputs):
        # The up_down_factor at the floor!
        up_down_f = 5
        # 一般假设 inputs的宽高是2**5的整数倍.
        down_x = inputs
        # Down！
        hor_x_0, down_x = down_rsu_with(self.rsu_d0, down_x)
        hor_x_1, down_x = down_rsu_with(self.rsu_d1, down_x)
        hor_x_2, down_x = down_rsu_with(self.rsu_d2, down_x)
        hor_x_3, down_x = down_rsu_with(self.rsu_d3, down_x)
        hor_x_4, down_x = down_rsu_with(self.rsu_d4, down_x)

        # Floor！
        side_floor, up_x = floor_rsu(self.rsu_f, down_x, up_down_f - 0, self.side_c)
        side_5 = side_floor

        # Up！
        side_4, up_x = up_rsu_with(self.rsu_u4, hor_x_4, up_x, up_down_f - 1, self.side_c)
        side_3, up_x = up_rsu_with(self.rsu_u3, hor_x_3, up_x, up_down_f - 2, self.side_c)
        side_2, up_x = up_rsu_with(self.rsu_u2, hor_x_2, up_x, up_down_f - 3, self.side_c)
        side_1, up_x = up_rsu_with(self.rsu_u1, hor_x_1, up_x, up_down_f - 4, self.side_c)
        side_0, up_x = up_rsu_with(self.rsu_u0, hor_x_0, up_x, up_down_f - 5, self.side_c)

        # Side！
        side_fuse = KL.concatenate([side_0, side_1, side_2, side_3, side_4, side_5])
        side_fuse = KL.Conv2D(self.side_c, kernel_size=3, padding='same')(side_fuse)
        side_fuse = KL.Activation('sigmoid', name='side_fuse')(side_fuse)

        side_5 = KL.Activation('sigmoid', name='side_5')(side_5)
        side_4 = KL.Activation('sigmoid', name='side_4')(side_4)
        side_3 = KL.Activation('sigmoid', name='side_3')(side_3)
        side_2 = KL.Activation('sigmoid', name='side_2')(side_2)
        side_1 = KL.Activation('sigmoid', name='side_1')(side_1)
        side_0 = KL.Activation('sigmoid', name='side_0')(side_0)

        # 拼到一起输出,应当也可以！！！
        side_out = KL.concatenate([side_fuse, side_0, side_1, side_2, side_3, side_4, side_5], name='side_all')
        return side_out

class U2netBuilder(object):
    @staticmethod
    def u_2_net(input_shape, side_c=1, name='U-2-NET'):
        x = KL.Input(shape=input_shape, name='image')
        side_out = U_2_Net(side_c=side_c)(x)

        model = KM.Model(inputs=x, outputs=side_out, name=name)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_fuse, side_0, side_1, side_2, side_3, side_4, side_5])
        return model

    @staticmethod
    def u_2_net_p(input_shape, side_c=1, name='U-2-NET-Simple'):
        x = KL.Input(shape=input_shape, name='image')
        side_out = U_2_Net(side_c=side_c, is_simple=True)(x)

        model = KM.Model(inputs=x, outputs=side_out, name=name)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_fuse, side_0, side_1, side_2, side_3, side_4, side_5])
        return model
