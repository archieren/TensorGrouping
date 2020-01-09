
#from tensorflow. keras.applications.resnet50 import ResNet50
#from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
#from keras.layers import ZeroPadding2D
#from keras.models import Model
#from keras.initializers import normal, constant, zeros
#from keras.regularizers import l2
#import keras.backend as K

from __future__ import division


import tensorflow as tf
from tensorflow import keras

KA = keras.applications
KL = keras.layers
KM = keras.models
KB = keras.backend
KU = keras.utils
KR = keras.regularizers

#from losses import loss
def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.math.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss


def loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return total_loss


def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


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
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
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


def decode( hm
            , wh
            , reg
            , max_objects=100
            , nms=True
            , flip_test=False
            , num_classes=20
            , score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],num_classes=num_classes,score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections

class CenterNetBuilder(object):
    @staticmethod
    def CenterNetOnResNet50V2(num_classes
                    #, backbone='resnet50'
                    , input_size=512
                    , max_objects=100
                    , score_threshold=0.1
                    , nms=True
                    , flip_test=False):
        #assert backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        output_size     = input_size // 4
        image_input     = KL.Input(shape=(32*16, 32*16, 3))
        hm_input        = KL.Input(shape=(output_size, output_size, num_classes))
        wh_input        = KL.Input(shape=(max_objects, 2))
        reg_input       = KL.Input(shape=(max_objects, 2))
        reg_mask_input  = KL.Input(shape=(max_objects,))
        index_input     = KL.Input(shape=(max_objects,))


        
        resnet = KA.ResNet50V2(weights='imagenet'
                            , input_tensor=image_input #KL.Input(shape=(32*16, 32*16, 3) # 32*outputshape = inputshape
                            , include_top=False)


        # (b, 16, 16, 2048)
        C5 = resnet.output
        #C5 = resnet.outputs[-1]


        x = KL.Dropout(rate=0.5)(C5)
        # decoder
        num_filters = [256, 128, 64]
        for nf in num_filters:
            x = KL.Conv2DTranspose(nf
                                , kernel_size=(4, 4)
                                , strides=2
                                , use_bias=False
                                , padding='same'
                                , kernel_initializer='he_normal'
                                , kernel_regularizer=KR.l2(5e-4))(x)
            x = KL.BatchNormalization()(x)
            x = KL.ReLU()(x)

        # hm header
        y1 = KL.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4))(x)
        y1 = KL.BatchNormalization()(y1)
        y1 = KL.ReLU()(y1)
        y1 = KL.Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4), activation='sigmoid')(y1)

        # wh header
        y2 = KL.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4))(x)
        y2 = KL.BatchNormalization()(y2)
        y2 = KL.ReLU()(y2)
        y2 = KL.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4))(y2)

        # reg header
        y3 = KL.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4))(x)
        y3 = KL.BatchNormalization()(y3)
        y3 = KL.ReLU()(y3)
        y3 = KL.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=KR.l2(5e-4))(y3)

        loss_ = KL.Lambda(loss, name='centernet_loss')(
            [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
        train_model = KM.Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

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