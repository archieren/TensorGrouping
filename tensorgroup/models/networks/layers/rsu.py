# See "U-2-Net:Going Deeper with Nested U-Structure for Salient Object Detection".
import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

def relu_bn_conv(filters=3, kernel_size=3, padding='same', dilation_rate=1):
    conv = KL.Conv2D(filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)
    bn = KL.BatchNormalization()
    relu = KL.Activation('relu')

    def f(x):
        x = relu(bn(conv(x)))
        return x

    return f

def down_RBC_with(rbc, down_in, pooling=True):
    hor_out = rbc(down_in)
    if pooling:
        down_out = KL.MaxPool2D(pool_size=2, strides=2)(hor_out)
    else:
        down_out = hor_out
    return hor_out, down_out

def up_RBC_with(rbc, hor_in, up_in, upsampling=True):
    up_out = KL.concatenate([hor_in, up_in])
    up_out = rbc(up_out)
    if upsampling:
        up_out = KL.UpSampling2D(size=2, interpolation='bilinear')(up_out)
    return up_out

def floor_RBC(rbc, down_in):
    up_out = rbc(down_in)
    return up_out

def residual(x, u_x):
    return KL.add([x, u_x])

def RSU7(filters=3, mid_filters=12):
    """
    The Residual U-Block 7
    Builds a custom residual u-net architecture!
    Args:
        filters: The output channels!
        mid_filters: The middle channels in the u-process!
    Returns:
        f: .
    """
    relu_bn_conv_d0 = relu_bn_conv(filters=filters)
    relu_bn_conv_d1 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d4 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d5 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d6 = relu_bn_conv(filters=mid_filters)

    relu_bn_conv_f = relu_bn_conv(filters=mid_filters, dilation_rate=2)

    relu_bn_conv_u6 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u5 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u4 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u1 = relu_bn_conv(filters=filters)

    def f(x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv_d0, down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv_d1, down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv_d2, down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv_d3, down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv_d4, down_x)
        hor_x_5, down_x = down_RBC_with(relu_bn_conv_d5, down_x)
        hor_x_6, down_x = down_RBC_with(relu_bn_conv_d6, down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv_f, down_x)

        up_x = up_RBC_with(relu_bn_conv_u6, hor_x_6, up_x)
        up_x = up_RBC_with(relu_bn_conv_u5, hor_x_5, up_x)
        up_x = up_RBC_with(relu_bn_conv_u4, hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv_u3, hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv_u2, hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv_u1, hor_x_1, up_x, upsampling=False)

        return residual(hor_x_0, up_x)

    return f

def RSU6(filters=3, mid_filters=12):
    relu_bn_conv_d0 = relu_bn_conv(filters=filters)
    relu_bn_conv_d1 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d4 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d5 = relu_bn_conv(filters=mid_filters)

    relu_bn_conv_f = relu_bn_conv(filters=mid_filters, dilation_rate=2)

    relu_bn_conv_u5 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u4 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u1 = relu_bn_conv(filters=filters)

    def f(x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv_d0, down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv_d1, down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv_d2, down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv_d3, down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv_d4, down_x)
        hor_x_5, down_x = down_RBC_with(relu_bn_conv_d5, down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv_f, down_x)

        up_x = up_RBC_with(relu_bn_conv_u5, hor_x_5, up_x)
        up_x = up_RBC_with(relu_bn_conv_u4, hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv_u3, hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv_u2, hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv_u1, hor_x_1, up_x, upsampling=False)

        return residual(hor_x_0, up_x)

    return f

def RSU5(filters=3, mid_filters=12):
    relu_bn_conv_d0 = relu_bn_conv(filters=filters)
    relu_bn_conv_d1 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d4 = relu_bn_conv(filters=mid_filters)

    relu_bn_conv_f = relu_bn_conv(filters=mid_filters, dilation_rate=2)

    relu_bn_conv_u4 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u1 = relu_bn_conv(filters=filters)

    def f(x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv_d0, down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv_d1, down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv_d2, down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv_d3, down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv_d4, down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv_f, down_x)

        up_x = up_RBC_with(relu_bn_conv_u4, hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv_u3, hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv_u2, hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv_u1, hor_x_1, up_x, upsampling=False)

        return residual(hor_x_0, up_x)

    return f

def RSU4(filters=3, mid_filters=12):
    relu_bn_conv_d0 = relu_bn_conv(filters=filters)
    relu_bn_conv_d1 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_d3 = relu_bn_conv(filters=mid_filters)

    relu_bn_conv_f = relu_bn_conv(filters=mid_filters, dilation_rate=2)

    relu_bn_conv_u3 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u2 = relu_bn_conv(filters=mid_filters)
    relu_bn_conv_u1 = relu_bn_conv(filters=filters)

    def f(x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv_d0, down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv_d1, down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv_d2, down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv_d3, down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv_f, down_x)

        up_x = up_RBC_with(relu_bn_conv_u3, hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv_u2, hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv_u1, hor_x_1, up_x, upsampling=False)

        return residual(hor_x_0, up_x)

    return f

def RSU4F(filters=3, mid_filters=12):
    relu_bn_conv_d0 = relu_bn_conv(filters=filters)
    relu_bn_conv_d1 = relu_bn_conv(filters=mid_filters, dilation_rate=1)
    relu_bn_conv_d2 = relu_bn_conv(filters=mid_filters, dilation_rate=2)
    relu_bn_conv_d3 = relu_bn_conv(filters=mid_filters, dilation_rate=4)

    relu_bn_conv_f = relu_bn_conv(filters=mid_filters, dilation_rate=8)

    relu_bn_conv_u3 = relu_bn_conv(filters=mid_filters, dilation_rate=4)
    relu_bn_conv_u2 = relu_bn_conv(filters=mid_filters, dilation_rate=2)
    relu_bn_conv_u1 = relu_bn_conv(filters=filters, dilation_rate=1)

    def f(x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv_d0, down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv_d1, down_x, pooling=False)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv_d2, down_x, pooling=False)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv_d3, down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv_f, down_x)

        up_x = up_RBC_with(relu_bn_conv_u3, hor_x_3, up_x, upsampling=False)
        up_x = up_RBC_with(relu_bn_conv_u2, hor_x_2, up_x, upsampling=False)
        up_x = up_RBC_with(relu_bn_conv_u1, hor_x_1, up_x, upsampling=False)

        return residual(hor_x_0, up_x)

    return f
