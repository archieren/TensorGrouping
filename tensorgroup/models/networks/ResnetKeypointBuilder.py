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

STRIDE = 4  # Net的Input Shape 和 Feature Map Shape的比

depth_switch = {None: RB.ResnetBuilder.build_resnet_50,
                "50": RB.ResnetBuilder.build_resnet_50,
                "101": RB.ResnetBuilder.build_resnet_101,
                "152": RB.ResnetBuilder.build_resnet_152}

def focal_loss(ck_hm_pred, ck_hm_true, ck_hm_mask):
    """
    Args: All with shape [B, Feature_Map_Shape, Num_Of_Classes]
       - ck_hm_pred: Predicated centerkeypoint heatmap.
       - ck_hm_true: Groundtruth of centerkeypoint heatmap
       - ck_hm_mask: Groundtruth of centerkeypoint center
    """
    pos_mask = tf.cast(ck_hm_mask, tf.float32)
    neg_mask = tf.cast(1-pos_mask, tf.float32)
    neg_weights = tf.pow(1 - ck_hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(ck_hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(1 - ck_hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - ck_hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(ck_hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices_pos, indices_mask):
    """
    Args:
       - y_pred: With shape [B, Feature_Map_Shape, 2]
       - y_true: With shape [B, N, 2]
       - indices_pos: With shape [B, N, 2]
       - indices_mask: With shape [B, N, 1]
    """
    y_pred = tf.gather_nd(y_pred, indices_pos, batch_dims=1)
    indices_mask = tf.tile(indices_mask, (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * indices_mask - y_pred * indices_mask))
    reg_loss = total_loss / (tf.reduce_sum(indices_mask) + 1e-4)
    return reg_loss

def loss(args):
    # [heatmap, center_offset, ck_hm_input, ck_mask_input, center_offset_input, indices_input, indices_mask_input]
    ck_hm_pred, center_offset_pred, ck_hm_true, ck_hm_mask, center_offset_true, indices_pos, indices_mask = args
    hm_loss = focal_loss(ck_hm_pred, ck_hm_true, ck_hm_mask)
    reg_loss = reg_l1_loss(center_offset_pred, center_offset_true, indices_pos, indices_mask)
    total_loss = hm_loss + reg_loss
    return tf.expand_dims(total_loss, axis=-1)


def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    b, _, w, _ = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]

    hm, indices_classid = tf.nn.top_k(hm, k=1)
    hm = tf.reshape(hm, (b, -1))

    scores, indices_pos = tf.nn.top_k(hm, k=max_objects)
    class_ids = tf.gather(tf.reshape(indices_classid, (b, -1)), indices_pos, batch_dims=1)
    xs = indices_pos % w
    ys = indices_pos // w
    indices_pos = ys * w + xs
    return scores, indices_pos, class_ids, xs, ys


def decode(hm, reg, max_objects=100, num_classes=2, score_threshold=0.1):
    scores, indices_pos, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices_pos, batch_dims=1)
    # (b, k, 2)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    # (b, k, 6)
    detections = tf.concat([topk_cx, topk_cy, scores, class_ids], axis=-1)
    return detections


class ResnetKeypointBuilder(object):
    @staticmethod
    def build(input_size,
              input_channels,
              num_classes,
              max_objects,
              score_threshold,
              resnet_depth=None):
        """Builds a custom Res_Keypoint_Net like architecture.
            See paper "Simple Baselines for Human Pose Estimation and Tracking"

        Args:
            - input_size, input_channels:
            - num_classes:
            - resnet_depth: The depth of the resnet!


        Returns:
            - The keras ResnetKeypoint `Model`.
        """
        output_size = input_size // STRIDE
        image_input = KL.Input(shape=(input_size, input_size, input_channels), name="image")
        ck_hm_input = KL.Input(shape=(output_size, output_size, num_classes), name='center_keypoint_heatmap')
        ck_mask_input = KL.Input(shape=(output_size, output_size, num_classes), name='center_keypoint_mask')
        center_offset_input = KL.Input(shape=(max_objects, 2), name='center_offset')
        indices_input = KL.Input(shape=(max_objects, 2), dtype=tf.int64, name='indices_pos')
        indices_mask_input = KL.Input(shape=(max_objects, 1), name='indices_mask')
        #
        if resnet_depth not in depth_switch:
            return None
        resnet_builder = depth_switch[resnet_depth]
        resnet = resnet_builder(image_input, 0, include_top=False)
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
        heatmap = KL.Conv2D(filters=num_classes, kernel_size=(1, 1), padding="same", use_bias=False, strides=1, name="heatmap_pred")(x)

        # reg header -- center_offset
        center_offset = KL.Conv2D(64, 3, padding='same')(x)
        center_offset = KL.BatchNormalization()(center_offset)
        center_offset = KL.ReLU()(center_offset)
        center_offset = KL.Conv2D(filters=2, kernel_size=(1, 1), padding="same", use_bias=False, strides=1, name="center_offset_pred")(center_offset)

        loss_ = KL.Lambda(loss, name='loss_as_output')([heatmap, center_offset, ck_hm_input, ck_mask_input, center_offset_input, indices_input, indices_mask_input])
        train_model = KM.Model(inputs=[image_input, indices_input, indices_mask_input, center_offset_input, ck_hm_input, ck_mask_input], outputs=[loss_])

        # detections = decode(y1, y2, y3)
        detections = KL.Lambda(lambda x: decode(*x, max_objects=max_objects, score_threshold=score_threshold, num_classes=num_classes))([heatmap, center_offset])
        prediction_model = KM.Model(inputs=image_input, outputs=detections)
        debug_model = KM.Model(inputs=image_input, outputs=[heatmap, center_offset])
        return train_model, prediction_model, debug_model

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
    def build_keypoint_resnet_50(input_size=512, input_channels=3, num_classes=2, max_objects=10, score_threshold=0.1):
        return ResnetKeypointBuilder.build(input_size, input_channels, num_classes, max_objects, score_threshold, resnet_depth="50")

    @staticmethod
    def build_keypoint_resnet_101(input_size=512, input_channels=3, num_classes=2, max_objects=10, score_threshold=0.1):
        return ResnetKeypointBuilder.build(input_size, input_channels, num_classes, max_objects, score_threshold, resnet_depth="101")

    @staticmethod
    def build_keypoint_resnet_152(input_size=512, input_channels=3, num_classes=2, max_objects=10, score_threshold=0.1):
        return ResnetKeypointBuilder.build(input_size, input_channels, num_classes, max_objects, score_threshold, resnet_depth="152")
