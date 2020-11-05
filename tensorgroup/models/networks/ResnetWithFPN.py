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
# 怕破坏原代码(ResnetBuilder),故拷过来,再改写。TODO:合并.

def _bn_relu(input):
    """Helper to build a BN -> relu block: relu(BN)
    """
    norm = KL.BatchNormalization(axis=CHANNEL_AXIS)(input)
    return KL.Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block: relu(BN(conv))
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", KR.l2(1.e-4))

    def f(input):
        conv = KL.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block: conv(relu(BN))
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", KR.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return KL.Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, padding=padding,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual, stride):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = KB.int_shape(input)
    residual_shape = KB.int_shape(residual)
    # stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    # stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    # 注意确保 stride == stride_width == stride_height。在Resnet环境下，可以做到的。
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride > 1 or not equal_channels:
        shortcut = KL.Conv2D(filters=residual_shape[CHANNEL_AXIS],
                             kernel_size=1,
                             strides=stride,
                             padding="valid",
                             kernel_initializer="he_normal",
                             kernel_regularizer=KR.l2(0.0001))(input)

    return KL.add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = 1
            if i == 0 and not is_first_layer:
                # 除去2、3、4层的0号残差块的初始stride为2，其他残差块的init_strides均为1
                init_strides = 2
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = KL.Conv2D(filters=filters, kernel_size=(3, 3),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=KR.l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual, stride=init_strides)

    return f


def bottleneck(filters, init_strides=1, is_first_block_of_first_layer=False):
    """
    Args:
        filters:
        init_strides: 其实确定了 input和residual的比例。
    Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = KL.Conv2D(filters=filters, kernel_size=1,
                                 strides=init_strides,
                                 padding="same",
                                 kernel_initializer="he_normal",
                                 kernel_regularizer=KR.l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=1,
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=3)(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=1)(conv_3_3)
        return _shortcut(input, residual, stride=init_strides)

    return f


class ResnetBuilder(object):
    @staticmethod
    def build(input, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input: Must have the shape as below!
                input_shape: The input shape in the form (nb_rows=32*??, nb_cols=32*??, nb_channels)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        # if len(input_shape) != 3:
        #     raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # input = KL.Input(shape=input_shape)
        stages = []
        P_filters = 256
        conv1 = _conv_bn_relu(filters=64, kernel_size=7, strides=2)(input)
        pool1 = KL.MaxPooling2D(pool_size=3, strides=2, padding="same")(conv1)

        block = pool1
        filters = 64
        stages.append(block)
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2
            # I don't sure whether _bn_relu is needed!但看了Keras Application中的实现,应当不用_bn_relu.
            # stages.append(_bn_relu(block))
            stages.append(block)

        # 其实, [C1, C2, C3, C4, C5] = stages[:5]
        fpn = ResnetBuilder.__gen_fpn(stages, P_filters)
        return fpn
        # -------------------------------------------------------------------
        # head4 = KL.Conv2D(P_filters // 2, 3, padding='same')(P5)
        # head4 = KL.Conv2D(P_filters // 2, 3, padding='same')(head4)

        # head3 = KL.Conv2D(P_filters // 2, 3, padding='same')(P4)
        # head3 = KL.Conv2D(P_filters // 2, 3, padding='same')(head3)

        # head2 = KL.Conv2D(P_filters // 2, 3, padding='same')(P3)
        # head2 = KL.Conv2D(P_filters // 2, 3, padding='same')(head2)

        # head1 = KL.Conv2D(P_filters // 2, 3, padding='same')(P2)
        # head1 = KL.Conv2D(P_filters // 2, 3, padding='same')(head1)

        # x = KL.Concatenate()([head1,
        #                       KL.UpSampling2D(size=2)(head2),
        #                       KL.UpSampling2D(size=4)(head3),
        #                       KL.UpSampling2D(size=8)(head4)])
        # x = KL.Conv2D(num_outputs, 3, padding='same', kernel_initializer='he_normal', activation='linear')(x)
        # x = KL.UpSampling2D(4)(x)
        # x = KL.Activation('sigmoid')(x)
        # model = KM.Model(inputs=input, outputs=x)
        # return model
        # -----------------------------------------------------------------------
        # # Last activation
        # block = _bn_relu(block)
        # if not include_top:
        #     model = KM.Model(inputs=input, outputs=block)
        #     return model

        # # Classifier block
        # block_shape = KB.int_shape(block)
        # pool2 = KL.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
        #                             strides=1)(block)
        # flatten1 = KL.Flatten()(pool2)
        # dense = KL.Dense(units=num_outputs, kernel_initializer="he_normal",
        #                  activation="softmax")(flatten1)

        # model = KM.Model(inputs=input, outputs=dense)

    @staticmethod
    def __gen_fpn(stages, filters):
        [C1, C2, C3, C4, C5] = stages[:5]
        # C1: [B, H/2, W/2, 64]
        # C2: [B, H/4, W/4, 256]
        # C3: [B, H/8, W/8, 512]
        # C4: [B, H/16, W/16, 1024]
        # C5: [B, H/32, W/32, 2048]

        # fpn_stride = [8, 16, 32, 64, 128]
        # P3: [B, H/8, W/8, 256]
        # P4: [B, H/16, W/16, 256]
        # P5: [B, H/32, W/32, 256]
        # P6: [B, H/64, W/64, 256]
        # P7: [B, H/128, W/128, 256]

        # upsample C5 to get P5 from the FPN paper
        P5 = KL.Conv2D(filters, 1)(C5)               # Hor_in flow
        P5_upsampled = KL.UpSampling2D(size=2)(P5)   # Down_out flow
        P5 = KL.Conv2D(filters, 3, padding='same', name='P5')(P5)  # 平滑 Hor_out flow

        P4 = KL.Conv2D(filters, 1)(C4)               # Hor_in flow
        P4 = KL.Add()([P5_upsampled, P4])
        P4_upsampled = KL.UpSampling2D(size=2)(P4)   # Down_out flow
        P4 = KL.Conv2D(filters, 3, padding='same', name='P4')(P4)  # 平滑 Hor_out flow

        P3 = KL.Conv2D(filters, 1)(C3)               # Hor_in flow
        P3 = KL.Add()([P4_upsampled, P3])
        # P3_upsampled = KL.UpSampling2D(size=2)(P3)   # Down_out flow
        P3 = KL.Conv2D(filters, 3, padding='same', name='P3')(P3)  # 平滑 Hor_out flow

        # P2 = KL.Conv2D(filters, 1)(C2)               # Hor_in flow
        # P2 = KL.Add()([P3_upsampled, P2])
        # P2_upsampled = KL.UpSampling2D(size=2)(P2)   # Down_out flow
        # P2 = KL.Conv2D(filters, 3, padding='same', name='P2')(P2)  # 平滑 Hor_out flow

        # P1 = KL.Conv2D(filters, 1)(C1)               # Hor_in flow
        # P1 = KL.Add()([P2_upsampled, P1])
        # P1 = KL.Conv2D(filters, 3, padding='same', name='P1')(P1)  # 平滑 Hor_out flow

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = KL.Conv2D(filters, 3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = KL.Activation('relu')(P6)
        P7 = KL.Conv2D(filters, 3, strides=2, padding='same', name='P7')(P7)

        return [P3, P4, P5, P6, P7]

    @staticmethod
    def build_resnet_18_fpn(input, num_outputs):
        return ResnetBuilder.build(input, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34_fpn(input, num_outputs):
        return ResnetBuilder.build(input, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50_fpn(input, num_outputs):
        return ResnetBuilder.build(input, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101_fpn(input, num_outputs):
        return ResnetBuilder.build(input, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152_fpn(input, num_outputs):
        return ResnetBuilder.build(input, num_outputs, bottleneck, [3, 8, 36, 3])
