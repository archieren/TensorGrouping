import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

class relu_bn_conv(KL.Layer):
    """
    The relu(bn(conv)) layer

    """

    def __init__(self, filters=3, kernel_size=3, padding='same', dilation_rate=1, **kwargs):
        super(relu_bn_conv, self).__init__(**kwargs)
        self.filters = filters
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size

    def build(self, input_shape):
        super(relu_bn_conv, self).build(input_shape)
        self.built = True

    def call(self, x):
        conv = KL.Conv2D(self.filters,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         dilation_rate=self.dilation_rate
                         )(x)
        bn = KL.BatchNormalization()(conv)
        relu = KL.Activation('relu')(bn)
        return relu

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


class RSU7(KL.Layer):
    """
    The Residual U-Block 7
    """

    def __init__(self, filters=3, mid_filters=12, **kwargs):
        """Builds a custom residual u-net architecture!

        Args:
            filters: The output channels!
            mid_filters: The middle channels in the u-process!
        Returns:
            The keras `Layer`.
        """
        super(RSU7, self).__init__(**kwargs)
        self.filters = filters
        self.mid_filters = mid_filters

    def build(self, input_shape):
        super(RSU7, self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv(filters=self.filters), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_5, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_6, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), down_x)

        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_6, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_5, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.filters), hor_x_1, up_x, upsampling=False)

        return hor_x_0+up_x

class RSU6(KL.Layer):
    """
    The Residual U-Block 6
    """

    def __init__(self, filters=3, mid_filters=12, **kwargs):
        """Builds a custom residual u-net architecture!

        Args:
            filters: The output channels!
            mid_filters: The middle channels in the u-process!
        Returns:
            The keras `Layer`.
        """
        super(RSU6, self).__init__(**kwargs)
        self.filters = filters
        self.mid_filters = mid_filters

    def build(self, input_shape):
        super(RSU6, self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv(filters=self.filters), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_5, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), down_x)

        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_5, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.filters), hor_x_1, up_x, upsampling=False)

        return hor_x_0+up_x


class RSU5(KL.Layer):
    """
    The Residual U-Block 5
    """

    def __init__(self, filters=3, mid_filters=12, **kwargs):
        """Builds a custom residual u-net architecture!

        Args:
            filters: The output channels!
            mid_filters: The middle channels in the u-process!
        Returns:
            The keras `Layer`.
        """
        super(RSU5, self).__init__(**kwargs)
        self.filters = filters
        self.mid_filters = mid_filters

    def build(self, input_shape):
        super(RSU5, self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv(filters=self.filters), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_4, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), down_x)

        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_4, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.filters), hor_x_1, up_x, upsampling=False)

        return hor_x_0+up_x

class RSU4(KL.Layer):
    """
    The Residual U-Block 4
    """

    def __init__(self, filters=3, mid_filters=12, **kwargs):
        """Builds a custom residual u-net architecture!

        Args:
            filters: The output channels!
            mid_filters: The middle channels in the u-process!
        Returns:
            The keras `Layer`.
        """
        super(RSU4, self).__init__(**kwargs)
        self.filters = filters
        self.mid_filters = mid_filters

    def build(self, input_shape):
        super(RSU4, self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv(filters=self.filters), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters), down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), down_x)

        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_3, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters), hor_x_2, up_x)
        up_x = up_RBC_with(relu_bn_conv(filters=self.filters, dilation_rate=1), hor_x_1, up_x, upsampling=False)

        return hor_x_0+up_x

class RSU4F(KL.Layer):
    """
    The Residual U-Block.
    The 'F' means flattened.
    """

    def __init__(self, filters=3, mid_filters=12, **kwargs):
        """Builds a custom residual u-net architecture!

        Args:
            filters: The output channels!
            mid_filters: The middle channels in the u-process!
        Returns:
            The keras `Layer`.
        """
        super(RSU4F, self).__init__(**kwargs)
        self.filters = filters
        self.mid_filters = mid_filters

    def build(self, input_shape):
        super(RSU4F, self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(relu_bn_conv(filters=self.filters, dilation_rate=1), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters, dilation_rate=1), down_x, pooling=False)
        hor_x_2, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), down_x, pooling=False)
        hor_x_3, down_x = down_RBC_with(relu_bn_conv(filters=self.mid_filters, dilation_rate=4), down_x, pooling=False)

        up_x = floor_RBC(relu_bn_conv(filters=self.mid_filters, dilation_rate=8), down_x)

        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters, dilation_rate=4), hor_x_3, up_x, upsampling=False)
        up_x = up_RBC_with(relu_bn_conv(filters=self.mid_filters, dilation_rate=2), hor_x_2, up_x, upsampling=False)
        up_x = up_RBC_with(relu_bn_conv(filters=self.filters, dilation_rate=1), hor_x_1, up_x, upsampling=False)

        return hor_x_0+up_x
