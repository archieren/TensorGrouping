

from __future__ import division


import tensorflow as tf
from tensorflow import keras

import tensorgroup.models.networks.ResnetBuilder as RB

KA = keras.applications
KL = keras.layers
KM = keras.models
KB = keras.backend
KU = keras.utils
KR = keras.regularizers

# from losses import loss


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
    ck_hm_pred, shape_pred, center_offset_pred, ck_hm_true, ck_hm_mask, shape_true, center_offset_true, indices_pos, indices_mask = args
    hm_loss = focal_loss(ck_hm_pred, ck_hm_true, ck_hm_mask)
    wh_loss = reg_l1_loss(shape_pred, shape_true, indices_pos, indices_mask)*0.1
    reg_loss = reg_l1_loss(center_offset_pred, center_offset_true, indices_pos, indices_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return tf.expand_dims(total_loss, axis=-1)


def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, _, w, _ = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]

    hm, indices_classid = tf.nn.top_k(hm, k=1)
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices_pos = tf.nn.top_k(hm, k=max_objects)
    class_ids = tf.gather(tf.reshape(indices_classid, (b, -1)), indices_pos, batch_dims=1)
    xs = indices_pos % w
    ys = indices_pos // w
    indices_pos = ys * w + xs
    return scores, indices_pos, class_ids, xs, ys


def evaluate_batch_item(batch_item_detections,
                        num_classes,
                        max_objects_per_class=20,
                        max_objects=100,
                        iou_threshold=0.5,
                        score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = KB.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = KB.concatenate(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices_pos = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices_pos)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm,
           wh,
           reg,
           max_objects=100,
           nms=True,
           flip_test=False,
           num_classes=20,
           score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices_pos, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices_pos, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices_pos, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 1:2] / 2
    topk_x2 = topk_cx + topk_wh[..., 1:2] / 2
    topk_y1 = topk_cy - topk_wh[..., 0:1] / 2
    topk_y2 = topk_cy + topk_wh[..., 0:1] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0], num_classes=num_classes, score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections


class CenterNetBuilder(object):
    @staticmethod
    def CenterNetOnResNet50V2(num_classes,  # backbone='resnet50'
                              input_size=512,  # 512 == 32*16
                              max_objects=100, score_threshold=0.1, nms=True, flip_test=False):
        # assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        # dataset 来的输入如下:（参见centernet_inputs.py）
        #        {'image': image,
        #         'indices_pos': indices_pos,
        #         'indices_mask': indices_mask,
        #         'center_offset': center_offset,
        #         'shape': shape,
        #         'center_keypoint_heatmap': center_keypoint_heatmap,
        #         'center_keypoint_mask': center_keypoint_mask}
        output_size = input_size // 4
        image_input = KL.Input(shape=(input_size, input_size, 3), name='image')
        ck_hm_input = KL.Input(shape=(output_size, output_size, num_classes), name='center_keypoint_heatmap')
        ck_mask_input = KL.Input(shape=(output_size, output_size, num_classes), name='center_keypoint_mask')
        shape_input = KL.Input(shape=(max_objects, 2), name='shape')
        center_offset_input = KL.Input(shape=(max_objects, 2), name='center_offset')
        indices_input = KL.Input(shape=(max_objects, 2), dtype=tf.int64, name='indices_pos')
        indices_mask_input = KL.Input(shape=(max_objects, 1), name='indices_mask')

        resnet = RB.ResnetBuilder.build_resnet_50(image_input, 0, include_top=True)
        # resnet.summary()
        # (b, 16, 16, 2048)
        resnet = resnet.output
        x = KL.Dropout(rate=0.5)(resnet)
        # 32*ResNetOutputSize = input_size
        # 生成feature_map
        x = CenterNetBuilder.make_deconv_layers(num_layers=3, num_filters=[256, 128, 64])(x)
        # (2**num_layers)*ResNetOutputSize = CenterNetOutputSize
        # hm header
        y1 = KL.Conv2D(64, 3, padding='same')(x)
        y1 = KL.BatchNormalization()(y1)
        y1 = KL.ReLU()(y1)
        y1 = KL.Conv2D(num_classes, 1, activation='sigmoid')(y1)

        # wh header -- shape
        y2 = KL.Conv2D(64, 3, padding='same')(x)
        y2 = KL.BatchNormalization()(y2)
        y2 = KL.ReLU()(y2)
        y2 = KL.Conv2D(2, 1)(y2)

        # reg header -- center_offset
        y3 = KL.Conv2D(64, 3, padding='same')(x)
        y3 = KL.BatchNormalization()(y3)
        y3 = KL.ReLU()(y3)
        y3 = KL.Conv2D(2, 1)(y3)

        loss_ = KL.Lambda(loss, name='loss_as_output')([y1, y2, y3, ck_hm_input, ck_mask_input, shape_input, center_offset_input, indices_input, indices_mask_input])
        train_model = KM.Model(inputs=[image_input, indices_input, indices_mask_input, center_offset_input, shape_input, ck_hm_input, ck_mask_input], outputs=[loss_])

        # detections = decode(y1, y2, y3)
        detections = KL.Lambda(lambda x: decode(*x,
                                                max_objects=max_objects,
                                                score_threshold=score_threshold,
                                                nms=nms,
                                                flip_test=flip_test,
                                                num_classes=num_classes))([y1, y2, y3])
        prediction_model = KM.Model(inputs=image_input, outputs=detections)
        debug_model = KM.Model(inputs=image_input, outputs=[y1, y2, y3])
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
