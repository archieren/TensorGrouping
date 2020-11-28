from __future__ import division


import tensorflow as tf
import tensorgroup.models.networks.ResnetBuilder as RB
KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

depth_switch = {None: RB.ResnetBuilder.build_resnet_50,
                "50": RB.ResnetBuilder.build_resnet_50,
                "101": RB.ResnetBuilder.build_resnet_101,
                "152": RB.ResnetBuilder.build_resnet_152}


class ResnetKeypointBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, resnet_depth=None):
        """Builds a custom Res_Keypoint_Net like architecture.
            See paper "Simple Baselines for Human Pose Estimation and Tracking"

        Args:
            - input_shape: The input shape in the form (nb_rows, nb_cols, nb_channels)
            - num_outputs: The number of outputs at final softmax layer
            - resnet_depth: The depth of the resnet!


        Returns:
            - The keras ResnetKeypoint `Model`.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        input = KL.Input(shape=input_shape, name="image")
        #
        if resnet_depth not in depth_switch:
            return None
        resnet_builder = depth_switch[resnet_depth]
        resnet = resnet_builder(input, 0, include_top=False)
        #
        x = resnet.output
        x = KL.Dropout(rate=0.5)(x)
        # 32*ResNetOutputSize = input_size
        x = ResnetKeypointBuilder.make_deconv_layers(num_layers=3, num_filters=[256, 128, 64])(x)
        # (2**num_layers)*ResNetOutputSize = ResnetKeypointOutputSize
        # heatmap
        heatmap = KL.Conv2D(64, 3, padding='same')(x)
        heatmap = KL.BatchNormalization()(heatmap)
        heatmap = KL.ReLU()(heatmap)
        heatmap = KL.Conv2D(filters=num_outputs, kernel_size=(1, 1), padding="same", use_bias=False, strides=1, name="heatmap")(x)

        # reg header -- center_offset
        center_offset = KL.Conv2D(64, 3, padding='same')(x)
        center_offset = KL.BatchNormalization()(center_offset)
        center_offset = KL.ReLU()(center_offset)
        center_offset = KL.Conv2D(filters=2, kernel_size=(1, 1), padding="same", use_bias=False, strides=1, name="center_offset")(center_offset)

        model = KM.Model(inputs=input, outputs=[heatmap, center_offset])
        return model

    @staticmethod
    def make_deconv_layers(num_layers=5, num_filters=[512, 256, 128, 64, 32]):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def f(input):
            for i in range(num_layers):
                nf = num_filters[i]
                # 1. Use a simple convolution instead of a deformable convolution
                input = KL.Conv2D(filters=nf, kernel_size=3, strides=1, padding='same')(input)
                input = KL.BatchNormalization()(input)
                input = KL.ReLU()(input)
                # 2. Use kernel_size=3, use_bias=True... which are different from oringinal kernel_size=4, use_bias=False...
                input = KL.Convolution2DTranspose(filters=nf, kernel_size=3, padding="same", strides=2)(input)
                input = KL.BatchNormalization()(input)
                input = KL.ReLU()(input)
            return input
        return f

    @staticmethod
    def build_keypoint_resnet_50(input_shape, num_outputs):
        return ResnetKeypointBuilder.build(input_shape, num_outputs, resnet_depth="50")

    @staticmethod
    def build_keypoint_resnet_101(input_shape, num_outputs):
        return ResnetKeypointBuilder.build(input_shape, num_outputs, resnet_depth="101")

    @staticmethod
    def build_keypoint_resnet_152(input_shape, num_outputs):
        return ResnetKeypointBuilder.build(input_shape, num_outputs, resnet_depth="152")
