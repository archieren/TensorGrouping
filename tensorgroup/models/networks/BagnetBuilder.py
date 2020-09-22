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

# 这个padding参数很奇怪，导致我总认为BagNet这篇论文有点在造数据！！？？
# padding=="valid",会导致tensor大小不匹配的情况发生.我干脆还是恢复padding="same"
_PAD = "same"


def _bottleneck(filters, is_k3=False, is_down=False):
    def f(input):
        # assert filters == KB.int_shape(input)[CHANNEL_AXIS]

        k3_size = 1 if not is_k3 else 3
        d_s = 2 if is_down else 1

        conv_1 = KL.Conv2D(filters, 1)(input)
        bn_1 = KL.BatchNormalization()(conv_1)
        relu_1 = KL.Activation("relu")(bn_1)

        conv_2 = KL.Conv2D(filters, k3_size, strides=d_s, padding=_PAD, use_bias=False)(relu_1)
        bn_2 = KL.BatchNormalization()(conv_2)
        relu_2 = KL.Activation("relu")(bn_2)

        conv_3 = KL.Conv2D(filters*4, 1, use_bias=False)(relu_2)
        bn_3 = KL.BatchNormalization()(conv_3)

        shortcut = KL.Conv2D(filters*4, 1, strides=d_s, padding=_PAD, use_bias=False)(input)
        shortcut_bn = KL.BatchNormalization()(shortcut)

        residual = KL.Add()([shortcut_bn, bn_3])
        relu_3 = KL.Activation("relu")(residual)

        return relu_3
    return f

def _build_layer(filters, repetition, is_k3=False, is_down=False):
    """Builds a residual block with repeating bottleneck blocks.
    is_kernel_3 means many things for the first block
    """
    def f(input):
        # 第一个瓶颈层,有些特殊,它的stride和kernel_size是要根据配置来调整的,而且和他的原型Resnet中对应的层处理的方式有差别！
        # 即kernel和stride的设置有区别！
        # 参见ResnetBuilder.py内的注释.
        input = _bottleneck(filters, is_k3=is_k3, is_down=is_down)(input)
        for _ in range(1, repetition):
            input = _bottleneck(filters)(input)
        return input

    return f

class BagnetBuilder(object):
    @staticmethod
    def build(input_shape, repetitions, k3, num_outputs):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
                一般来理解的话，定义时用(None, None, 3),训练时用(224, 224, 3), 推理时，指定(9, 9, 3) 、(17, 17, 3) 、(33, 33, 3)。
            num_outputs: The number of outputs at final softmax layer
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
            k3: 各层是否使用大小为3的卷积核。
            num_outputs: 类别数。
        Returns:
            The keras `Model`.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_rows, nb_cols, nb_channels)")
        assert len(repetitions) == len(k3), 'ERROR: len(repetitions) is different len(k3)'

        input = KL.Input(shape=input_shape)

        conv0 = KL.Conv2D(64, 1)(input)
        conv1 = KL.Conv2D(64, 3, padding=_PAD)(conv0)
        bn = KL.BatchNormalization()(conv1)
        relu = KL.Activation("relu")(bn)
        prep = relu

        layer0 = _build_layer(64, repetitions[0], is_k3=True if k3[0] == 1 else False, is_down=True)(prep)
        layer1 = _build_layer(128, repetitions[1], is_k3=True if k3[1] == 1 else False, is_down=True)(layer0)
        layer2 = _build_layer(256, repetitions[2], is_k3=True if k3[2] == 1 else False, is_down=True)(layer1)
        layer3 = _build_layer(512, repetitions[3], is_k3=True if k3[3] == 1 else False, is_down=False)(layer2)

        # Classifier block
        # 对于BagNet,我始终没明白的是按224训练,按9\17\33来使用,是如何用下面的方法做到的.
        # 故我将它换成GlobalAvgPool2D以适应输入形式的可变性. 看其源代码,不过也应当是这种意思.
        # block_shape = KB.int_shape(block)
        # pool2 = KL.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(block)
        # flatten1 = KL.Flatten(pool2)
        pool = KL.GlobalAvgPool2D()(layer3)
        dense = KL.Dense(units=num_outputs, kernel_initializer="he_normal",
                         activation="softmax")(pool)

        model = KM.Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_bagnet_N(n=4, num_outputs=1000):  # input_shape=(2**n+1, 2**n+1, 3) n=3, 4, 5
        assert n > 2
        size = 2**n+1
        input_shape = (size, size, 3)
        repetitions = [3, 4, 5, 6]
        k3 = [0, 0, 0, 0]
        for i in range(4):
            k3[i] = 1 if i < n-1 else 0
        return BagnetBuilder.build(input_shape, repetitions, k3, num_outputs)
