# See "U-2-Net:Going Deeper with Nested U-Structure for Salient Object Detection".
import tensorflow as tf  # TF 2.0
import tensorgroup.models.networks.layers.rsu as RSU

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

"""
解决"ValueError: tf.function-decorated function tried to create variables on non-first call."的方法。
对于含有变量的层，不能放到函数里来处理。只能采用一次定义，后引用的模式。
"""


def down_rsu_with(rsu, pooling=True):
    maxpool = KL.MaxPool2D(pool_size=2, strides=2)

    def f(down_in):
        hor_out = rsu(down_in)
        if pooling:
            down_out = maxpool(hor_out)
        else:
            down_out = hor_out
        return hor_out, down_out
    return f

def up_rsu_with(rsu, up_down_scale, side_c, upsampling=True):
    conv = KL.Conv2D(side_c, kernel_size=3, padding='same')
    up_us = KL.UpSampling2D(size=2, interpolation='bilinear')
    side_us = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')

    def f(hor_in, up_in):
        if upsampling:
            up_out = up_us(up_in)
        else:
            up_out = up_in
        up_out = KL.concatenate([hor_in, up_out])
        up_out = rsu(up_out)
        side_fuse = conv(up_out)
        if up_down_scale > 0:
            side_fuse = side_us(side_fuse)
        return side_fuse, up_out

    return f

def floor_rsu(rsu, up_down_scale, side_c):
    conv = KL.Conv2D(side_c, kernel_size=3, padding='same')
    side_us = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')

    def f(down_in):
        up_out = rsu(down_in)
        side_fuse = conv(up_out)
        if up_down_scale > 0:
            side_fuse = side_us(side_fuse)
        return side_fuse, up_out
    return f

def U_2_Net(side_c=1, is_simple=False, output_name='side_all'):
    if not is_simple:
        rsu_d0 = RSU.RSU7(filters=64, mid_filters=32)
        rsu_d1 = RSU.RSU6(filters=128, mid_filters=32)
        rsu_d2 = RSU.RSU5(filters=256, mid_filters=64)
        rsu_d3 = RSU.RSU4(filters=512, mid_filters=128)
        rsu_d4 = RSU.RSU4F(filters=512, mid_filters=256)

        rsu_fl = RSU.RSU4F(filters=512, mid_filters=256)

        rsu_u4 = RSU.RSU4F(filters=512, mid_filters=256)
        rsu_u3 = RSU.RSU4(filters=256, mid_filters=128)
        rsu_u2 = RSU.RSU5(filters=128, mid_filters=64)
        rsu_u1 = RSU.RSU6(filters=64, mid_filters=32)
        rsu_u0 = RSU.RSU7(filters=64, mid_filters=16)
    else:
        rsu_d0 = RSU.RSU7(filters=64, mid_filters=16)
        rsu_d1 = RSU.RSU6(filters=64, mid_filters=16)
        rsu_d2 = RSU.RSU5(filters=64, mid_filters=16)
        rsu_d3 = RSU.RSU4(filters=64, mid_filters=16)
        rsu_d4 = RSU.RSU4F(filters=64, mid_filters=16)

        rsu_fl = RSU.RSU4F(filters=64, mid_filters=16)

        rsu_u4 = RSU.RSU4F(filters=64, mid_filters=16)
        rsu_u3 = RSU.RSU4(filters=64, mid_filters=16)
        rsu_u2 = RSU.RSU5(filters=64, mid_filters=16)
        rsu_u1 = RSU.RSU6(filters=64, mid_filters=16)
        rsu_u0 = RSU.RSU7(filters=64, mid_filters=16)

    # The up_down_factor at the floor!
    up_down_f = 5
    down0 = down_rsu_with(rsu_d0)
    down1 = down_rsu_with(rsu_d1)
    down2 = down_rsu_with(rsu_d2)
    down3 = down_rsu_with(rsu_d3)
    down4 = down_rsu_with(rsu_d4)
    floor = floor_rsu(rsu_fl, up_down_f - 0, side_c)
    up4 = up_rsu_with(rsu_u4, up_down_f - 1, side_c)
    up3 = up_rsu_with(rsu_u3, up_down_f - 2, side_c)
    up2 = up_rsu_with(rsu_u2, up_down_f - 3, side_c)
    up1 = up_rsu_with(rsu_u1, up_down_f - 4, side_c)
    up0 = up_rsu_with(rsu_u0, up_down_f - 5, side_c)

    def f(inputs):
        # The up_down_factor at the floor!
        up_down_f = 5
        # 一般假设 inputs的宽高是2**5的整数倍.
        down_x = inputs
        # Down！
        hor_x_0, down_x = down0(down_x)
        hor_x_1, down_x = down1(down_x)
        hor_x_2, down_x = down2(down_x)
        hor_x_3, down_x = down3(down_x)
        hor_x_4, down_x = down4(down_x)

        # Floor！
        side_floor, up_x = floor(down_x)
        side_5 = side_floor

        # Up！
        side_4, up_x = up4(hor_x_4, up_x)
        side_3, up_x = up3(hor_x_3, up_x)
        side_2, up_x = up2(hor_x_2, up_x)
        side_1, up_x = up1(hor_x_1, up_x)
        side_0, up_x = up0(hor_x_0, up_x)

        # Side！
        side_fuse = KL.concatenate([side_0, side_1, side_2, side_3, side_4, side_5])
        side_fuse = KL.Conv2D(side_c, kernel_size=1, padding='same')(side_fuse)  # kernel_size=3
        side_fuse = KL.Activation('sigmoid', name='side_fuse')(side_fuse)

        if output_name != 'side_fuse':
            side_5 = KL.Activation('sigmoid', name='side_5')(side_5)
            side_4 = KL.Activation('sigmoid', name='side_4')(side_4)
            side_3 = KL.Activation('sigmoid', name='side_3')(side_3)
            side_2 = KL.Activation('sigmoid', name='side_2')(side_2)
            side_1 = KL.Activation('sigmoid', name='side_1')(side_1)
            side_0 = KL.Activation('sigmoid', name='side_0')(side_0)

            # 拼到一起输出,应当也可以！！！
            side_out = KL.concatenate([side_fuse, side_0, side_1, side_2, side_3, side_4, side_5], name=output_name)
        else:
            side_out = side_fuse
        return side_out

    return f

class U2netBuilder(object):
    @staticmethod
    def u_2_net(input_shape, side_c=1, name='U-2-NET', output_name='side_all'):
        x = KL.Input(shape=input_shape, name='image')
        side_out = U_2_Net(side_c=side_c, is_simple=False, output_name=output_name)(x)

        model = KM.Model(inputs=x, outputs=side_out, name=name)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_fuse, side_0, side_1, side_2, side_3, side_4, side_5])
        return model

    @staticmethod
    def u_2_net_p(input_shape, side_c=1, name='U-2-NET-Simple', output_name='side_all'):
        x = KL.Input(shape=input_shape, name='image')
        side_out = U_2_Net(side_c=side_c, is_simple=True, output_name=output_name)(x)

        model = KM.Model(inputs=x, outputs=side_out, name=name)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_fuse, side_0, side_1, side_2, side_3, side_4, side_5])
        return model
