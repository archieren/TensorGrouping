from __future__ import division


import tensorflow as tf

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

# 有几件事需要说一下：
# 1) 残差网中,padding用的都是"same",而不是这的"valid"
# 2) 修改了shortcut里的内容，参照作者pytorch的实现！

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = KL.BatchNormalization(axis=CHANNEL_AXIS)(input)
    return KL.Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "valid")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", KR.l2(1.e-4))

    def f(input):
        conv = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "valid")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", KR.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return KL.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = KB.int_shape(input)
    residual_shape = KB.int_shape(residual)
    # print("{}.{}".format(input_shape,residual_shape))
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # In the first bottleneck of each _residual_block, this will be in case !
    # 每一残差块的第一个瓶颈层，会出现这种情况？
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = KL.Conv2D(filters=residual_shape[CHANNEL_AXIS],
                             kernel_size=(1, 1),
                             strides=(stride_width, stride_height),
                             padding="valid",
                             kernel_initializer="he_normal",
                             kernel_regularizer=KR.l2(0.0001))(input)
        shortcut = shortcut[:, :residual_shape[ROW_AXIS], :residual_shape[ROW_AXIS], :]
    # print(shortcut.shape)
    # print(residual.shape)
    return KL.add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_kernel_3, k3_stride):
    """Builds a residual block with repeating bottleneck blocks.
    is_kernel_3 means many things for the first block
    """
    def f(input):
        # 第一个瓶颈层,有些特殊,它的stride和kernel_size是要根据配置来调整的,而且和他的原型Resnet中对应的层处理的方式有差别！
        # 即kernel和stride的设置有区别！
        # 参见ResnetBuilder.py内的注释.
        input = block_function(filters, k3_stride=k3_stride, is_kernel_3=is_kernel_3)(input)
        for _ in range(1, repetitions):
            input = block_function(filters=filters, k3_stride=1, is_kernel_3=False)(input)
        return input

    return f


def _conv_bn_residual_relu(**conv_params):
    """Helper to build a conv -> BN ->Residual -> Relu block
    这是我根据Bagnet里的要求改的！在Bagnet里，residual在Relu前.
    """
    init_input = conv_params["input"]
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "valid")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", KR.l2(1.e-4))

    def f(input):
        conv = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(input)
        norm = KL.BatchNormalization(axis=CHANNEL_AXIS)(conv)
        residual = _shortcut(init_input, norm)

        return KL.Activation("relu")(residual)

    return f

def bottleneck(filters, is_kernel_3=False, k3_stride=1):
    """
    The second _conv_bn_relu's kernel_size is determined by is_kernel_3
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):
        if is_kernel_3:
            k3_size = 3
        else:
            k3_size = 1

        conv_1_1 = _conv_bn_relu(filters=filters, kernel_size=(1, 1))(input)
        conv_3_3 = _conv_bn_relu(filters=filters, kernel_size=(k3_size, k3_size), strides=(k3_stride, k3_stride))(conv_1_1)
        residual = _conv_bn_residual_relu(input=input, filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return residual  # _shortcut(input, residual)

    return f


class BagnetBuilder(object):
    @staticmethod
    def build(input_shape, repetitions, k3, strides, num_outputs):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
            num_outputs: The number of outputs at final softmax layer
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        block_fn = bottleneck
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
        assert len(repetitions) == len(k3), 'ERROR: len(repetitions) is different len(k3)'
        assert len(repetitions) == len(strides), 'ERROR: len(repetitions) is different len(strides)'

        input = KL.Input(shape=input_shape)
        conv0 = KL.Conv2D(filters=64,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=KR.l2(1.e-4))(input)

        conv1 = _conv_bn_relu(filters=64,
                              kernel_size=(3, 3),
                              padding="valid")(conv0)

        block = conv1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn,
                                    filters=filters,
                                    repetitions=r,
                                    is_kernel_3=True if k3[i] == 1 else False,
                                    k3_stride=strides[i]
                                    )(block)
            filters *= 2

        # Last activation
        # block = _bn_relu(block) # 没必要了！

        # Classifier block
        block_shape = KB.int_shape(block)
        pool2 = KL.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                    strides=(1, 1))(block)
        flatten1 = KL.Flatten()(pool2)
        dense = KL.Dense(units=num_outputs, kernel_initializer="he_normal",
                         activation="softmax")(flatten1)

        model = KM.Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_bagnet_9(input_shape=(9, 9, 3), num_outputs=1000):
        return BagnetBuilder.build(input_shape,
                                   # bottleneck,
                                   repetitions=[3, 4, 6, 3],
                                   k3=[1, 1, 0, 0],
                                   strides=[2, 2, 2, 1],
                                   num_outputs=num_outputs)

    @staticmethod
    def build_bagnet_17(input_shape=(17, 17, 3), num_outputs=1000):
        return BagnetBuilder.build(input_shape,
                                   # bottleneck,
                                   repetitions=[3, 4, 23, 3],
                                   k3=[1, 1, 1, 0],
                                   strides=[2, 2, 2, 1],
                                   num_outputs=num_outputs)

    @staticmethod
    def build_bagnet_33(input_shape=(33, 33, 3), num_outputs=1000):
        return BagnetBuilder.build(input_shape,
                                   # bottleneck,
                                   repetitions=[3, 8, 36, 3],
                                   k3=[1, 1, 1, 1],
                                   strides=[2, 2, 2, 1],
                                   num_outputs=num_outputs)
