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
    side_out = KL.Conv2D(side_c, kernel_size=3, padding='same')(up_out)
    if up_down_scale > 0:
        side_out = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')(side_out)
    return side_out, up_out

def floor_rsu(rsu, down_in, up_down_scale, side_c):
    up_out = rsu(down_in)
    side_out = KL.Conv2D(side_c, kernel_size=3, padding='same')(up_out)
    if up_down_scale > 0:
        side_out = KL.UpSampling2D(size=2**up_down_scale, interpolation='bilinear')(side_out)
    return side_out, up_out

class U2netBuilder(object):
    @staticmethod
    def u_2_net(input_shape, side_c=1):
        # The up_down_factor at the floor!
        up_down_f = 5

        x = KL.Input(shape=input_shape, name='image')
        down_x = x
        # Down！
        hor_x_0, down_x = down_rsu_with(RSU.RSU7(filters=64, mid_filters=32), down_x)
        hor_x_1, down_x = down_rsu_with(RSU.RSU6(filters=128, mid_filters=32), down_x)
        hor_x_2, down_x = down_rsu_with(RSU.RSU5(filters=256, mid_filters=64), down_x)
        hor_x_3, down_x = down_rsu_with(RSU.RSU4(filters=512, mid_filters=128), down_x)
        hor_x_4, down_x = down_rsu_with(RSU.RSU4F(filters=512, mid_filters=256), down_x)

        # Floor！
        side_floor, up_x = floor_rsu(RSU.RSU4F(filters=512, mid_filters=256), down_x, up_down_f - 0, side_c)
        side_5 = side_floor

        # Up！
        side_4, up_x = up_rsu_with(RSU.RSU4F(filters=512, mid_filters=256), hor_x_4, up_x, up_down_f - 1, side_c)
        side_3, up_x = up_rsu_with(RSU.RSU4(filters=256, mid_filters=126), hor_x_3, up_x, up_down_f - 2, side_c)
        side_2, up_x = up_rsu_with(RSU.RSU5(filters=128, mid_filters=64), hor_x_2, up_x, up_down_f - 3, side_c)
        side_1, up_x = up_rsu_with(RSU.RSU6(filters=64, mid_filters=32), hor_x_1, up_x, up_down_f - 4, side_c)
        side_0, up_x = up_rsu_with(RSU.RSU7(filters=64, mid_filters=16), hor_x_0, up_x, up_down_f - 5, side_c)

        # Side！
        side_out = KL.concatenate([side_0, side_1, side_2, side_3, side_4, side_5])
        side_out = KL.Conv2D(side_c, kernel_size=3, padding='same')(side_out)
        side_out = KL.Activation('sigmoid', name='side_out')(side_out)

        side_5 = KL.Activation('sigmoid', name='side_5')(side_5)
        side_4 = KL.Activation('sigmoid', name='side_4')(side_4)
        side_3 = KL.Activation('sigmoid', name='side_3')(side_3)
        side_2 = KL.Activation('sigmoid', name='side_2')(side_2)
        side_1 = KL.Activation('sigmoid', name='side_1')(side_1)
        side_0 = KL.Activation('sigmoid', name='side_0')(side_0)

        # 拼到一起输出,应当也可以！！！
        side_out = KL.concatenate([side_out, side_0, side_1, side_2, side_3, side_4, side_5], name='side_all')
        model = KM.Model(inputs=x, outputs=side_out)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_out, side_0, side_1, side_2, side_3, side_4, side_5])
        return model

    @staticmethod
    def u_2_net_p(input_shape, side_c=1):
        # The up_down_factor at the floor!
        up_down_f = 5

        x = KL.Input(shape=input_shape, name='image')
        down_x = x

        # Down flow！
        hor_x_0, down_x = down_rsu_with(RSU.RSU7(filters=64, mid_filters=16), down_x)
        hor_x_1, down_x = down_rsu_with(RSU.RSU6(filters=64, mid_filters=16), down_x)
        hor_x_2, down_x = down_rsu_with(RSU.RSU5(filters=64, mid_filters=16), down_x)
        hor_x_3, down_x = down_rsu_with(RSU.RSU4(filters=64, mid_filters=16), down_x)
        hor_x_4, down_x = down_rsu_with(RSU.RSU4F(filters=64, mid_filters=16), down_x)

        # Floor flow！
        side_floor, up_x = floor_rsu(RSU.RSU4F(filters=64, mid_filters=16), down_x, up_down_f - 0, side_c)
        side_5 = side_floor  # Just for reading!!!

        # Up flow！
        side_4, up_x = up_rsu_with(RSU.RSU4F(filters=64, mid_filters=16), hor_x_4, up_x, up_down_f - 1, side_c)
        side_3, up_x = up_rsu_with(RSU.RSU4(filters=64, mid_filters=16), hor_x_3, up_x, up_down_f - 2, side_c)
        side_2, up_x = up_rsu_with(RSU.RSU5(filters=64, mid_filters=16), hor_x_2, up_x, up_down_f - 3, side_c)
        side_1, up_x = up_rsu_with(RSU.RSU6(filters=64, mid_filters=16), hor_x_1, up_x, up_down_f - 4, side_c)
        side_0, up_x = up_rsu_with(RSU.RSU7(filters=64, mid_filters=16), hor_x_0, up_x, up_down_f - 5, side_c)

        # Side
        side_out = KL.concatenate([side_0, side_1, side_2, side_3, side_4, side_5])
        side_out = KL.Conv2D(side_c, kernel_size=3, padding='same')(side_out)
        side_out = KL.Activation('sigmoid', name='side_out')(side_out)

        side_5 = KL.Activation('sigmoid', name='side_5')(side_5)
        side_4 = KL.Activation('sigmoid', name='side_4')(side_4)
        side_3 = KL.Activation('sigmoid', name='side_3')(side_3)
        side_2 = KL.Activation('sigmoid', name='side_2')(side_2)
        side_1 = KL.Activation('sigmoid', name='side_1')(side_1)
        side_0 = KL.Activation('sigmoid', name='side_0')(side_0)

        # 拼到一起输出,应当也可以！！！
        side_out = KL.concatenate([side_out, side_0, side_1, side_2, side_3, side_4, side_5], name='side_all')
        model = KM.Model(inputs=x, outputs=side_out)

        # MultiOutputs,这是作者的方式！
        # model = KM.Model(inputs=x, outputs=[side_out, side_0, side_1, side_2, side_3, side_4, side_5])
        return model
