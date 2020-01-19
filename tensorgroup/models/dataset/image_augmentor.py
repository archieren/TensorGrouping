import tensorflow as tf
#import tensorflow_addons as tfa


def image_augmentor(image
                    #, input_shape
                    , data_format 
                    , output_shape
                    , zoom_size=None
                    , crop_method=None
                    , flip_prob=None
                    , fill_mode='BILINEAR'
                    , keep_aspect_ratios=False
                    , constant_values=0.
                    , color_jitter_prob=None
                    , rotate=None
                    , ground_truth=None
                    , pad_truth_to=None):

    """
    Args:
        :param image: HWC or CHW
        :param input_shape: [h, w]
        :param data_format: 'channels_first', 'channels_last'
        :param output_shape: [h, w]
        :param zoom_size: [h, w]
        :param crop_method: 'random', 'center'
        :param flip_prob: [flip_top_down_prob, flip_left_right_prob]
        :param fill_mode: 'CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC'
        :param keep_aspect_ratios: True, False
        :param constant_values:
        :param color_jitter_prob: prob of color_jitter
        :param rotate: [prob, min_angle, max_angle]
        :param ground_truth: [ymin, ymax, xmin, xmax, classid]
        :param pad_truth_to: pad ground_truth to size [pad_truth_to, 5] with -1
    Rerurns:
        :return image: output_shape
        :return ground_truth: [pad_truth_to, 5] [ycenter, xcenter, h, w, class_id]

    """

    if data_format not in ['channels_first', 'channels_last']:
        raise Exception("data_format must in ['channels_first', 'channels_last']!")
    if fill_mode not in ['CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC']:
        raise Exception("fill_mode must in ['CONSTANT', 'NEAREST_NEIGHBOR', 'BILINEAR', 'BICUBIC']!")
    if fill_mode == 'CONSTANT' and zoom_size is not None:
        raise Exception("if fill_mode is 'CONSTANT', zoom_size can't be None!")
    if zoom_size is not None:
        if keep_aspect_ratios:
            if constant_values is None:
                raise Exception('please provide constant_values!')
        if not zoom_size[0] >= output_shape[0] and zoom_size[1] >= output_shape[1]:
            raise Exception("output_shape can't greater that zoom_size!")
        if crop_method not in ['random', 'center']:
            raise Exception("crop_method must in ['random', 'center']!")
        if fill_mode is 'CONSTANT' and constant_values is None:
            raise Exception("please provide constant_values!")
    if color_jitter_prob is not None:
        if not 0. <= color_jitter_prob <= 1.:
            raise Exception("color_jitter_prob can't less that 0.0, and can't grater that 1.0")
    if flip_prob is not None:
        if not 0. <= flip_prob[0] <= 1. and 0. <= flip_prob[1] <= 1.:
            raise Exception("flip_prob can't less than 0.0, and can't grater than 1.0")
    if rotate is not None:
        if len(rotate) != 3:
            raise Exception('please provide "rotate" parameter as [rotate_prob, min_angle, max_angle]!')
        if not 0. <= rotate[0] <= 1.:
            raise Exception("rotate prob can't less that 0.0, and can't grater that 1.0")
        if ground_truth is not None:
            if not -5. <= rotate[1] <= 5. and -5. <= rotate[2] <= 5.:
                raise Exception('rotate range must be -5 to 5, otherwise coordinate mapping become imprecise!')
        if not rotate[1] <= rotate[2]:
            raise Exception("rotate[1] can't  grater than rotate[2]")


    image = tf.image.convert_image_dtype(image,tf.float32)
    input_shape = tf.shape(image)
    if data_format == 'channels_first':
        image = tf.transpose(image, [1, 2, 0])
    input_h, input_w, input_c = input_shape[0], input_shape[1], input_shape[2]
    output_h, output_w = output_shape    

    if fill_mode == 'CONSTANT':  #如果填充区为常量，则需保持 aspect_ratios
        keep_aspect_ratios = True
    fill_mode_project = {
        'NEAREST_NEIGHBOR': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'BILINEAR': tf.image.ResizeMethod.BILINEAR,
        'BICUBIC': tf.image.ResizeMethod.BICUBIC
    }
            
    image = tf.image.resize(image
                                , [output_h, output_w]
                                , method=fill_mode_project[fill_mode]
                                , preserve_aspect_ratio=False
    )
    image = color_jitter(image, color_jitter_prob)
    image , ground_truth = flip(image, ground_truth,flip_prob)
    ground_truth =  filter_ground_truth(ground_truth)
    ground_truth = yyxx_to_yxhw(ground_truth)
    ground_truth =  pad_truth(ground_truth,pad_truth_to)
    return image, ground_truth

    

def flip(image,ground_truth,flip_prob):
    ymin = tf.reshape(ground_truth[:, 0], [-1, 1])
    ymax = tf.reshape(ground_truth[:, 1], [-1, 1])
    xmin = tf.reshape(ground_truth[:, 2], [-1, 1])
    xmax = tf.reshape(ground_truth[:, 3], [-1, 1])
    class_id = tf.reshape(ground_truth[:, 4], [-1, 1])

    flip_td_prob = tf.random.uniform([], 0., 1.)
    flip_lr_prob = tf.random.uniform([], 0., 1.)
    image = tf.cond(tf.less(flip_td_prob, flip_prob[0]),lambda: tf.reverse(image, [0]),lambda: image)
    image = tf.cond(tf.less(flip_lr_prob, flip_prob[1]),lambda: tf.reverse(image, [1]),lambda: image)
    ymax, ymin = tf.cond(tf.less(flip_td_prob, flip_prob[0]),lambda: (1.-ymin , 1.-ymax),lambda: (ymax, ymin))
    xmax, xmin = tf.cond(tf.less(flip_lr_prob, flip_prob[1]),lambda: (1. - xmin, 1. - xmax),lambda: (xmax, xmin))  
    
    ground_truth = tf.concat([ymin,ymax,xmin,xmax,class_id], axis=-1)
    return image, ground_truth  

def color_jitter(image, color_jitter_prob):
    if color_jitter_prob is not None:
        bcs = tf.random.uniform([3], 0., 1.)
        image = tf.cond(bcs[0] < color_jitter_prob,
                        lambda: tf.image.adjust_brightness(image, tf.random.uniform([], 0., 0.3)),
                        lambda: image
                )
        image = tf.cond(bcs[1] < color_jitter_prob,
                        lambda: tf.image.adjust_contrast(image, tf.random.uniform([], 0.8, 1.2)),
                        lambda: image
                )
        image = tf.cond(bcs[2] < color_jitter_prob,
                        lambda: tf.image.adjust_hue(image, tf.random.uniform([], -0.1, 0.1)),
                        lambda: image
                )    
    return image

def filter_ground_truth(ground_truth):
    ymin = tf.reshape(ground_truth[:, 0], [-1, 1])
    ymax = tf.reshape(ground_truth[:, 1], [-1, 1])
    xmin = tf.reshape(ground_truth[:, 2], [-1, 1])
    xmax = tf.reshape(ground_truth[:, 3], [-1, 1])
    class_id = tf.reshape(ground_truth[:, 4], [-1, 1])

    y_center = (ymin + ymax) / 2.
    x_center = (xmin + xmax) / 2.
    y_mask = tf.cast(y_center > 0., tf.float32) * tf.cast(y_center < 1., tf.float32)
    x_mask = tf.cast(x_center > 0., tf.float32) * tf.cast(x_center < 1., tf.float32)
    mask = tf.reshape((x_mask * y_mask) > 0., [-1])
    ymin = tf.boolean_mask(ymin, mask)
    xmin = tf.boolean_mask(xmin, mask)
    ymax = tf.boolean_mask(ymax, mask)
    xmax = tf.boolean_mask(xmax, mask)
    class_id = tf.boolean_mask(class_id, mask)
    ymin = tf.where(ymin < 0., 0., ymin)
    xmin = tf.where(xmin < 0., 0., xmin)
    ymax = tf.where(ymax < 0., 0., ymax)
    xmax = tf.where(xmax < 0., 0., xmax)
    ymin = tf.where(ymin > 1., 1., ymin)
    xmin = tf.where(xmin > 1., 1., xmin)
    ymax = tf.where(ymax > 1., 1., ymax)
    xmax = tf.where(xmax > 1., 1., xmax)
    ground_truth_= tf.concat([ymin, ymax, xmin, xmax, class_id], axis=-1)
    
    

    return ground_truth_

def yyxx_to_yxhw(ground_truth):
    ymin = tf.reshape(ground_truth[:, 0], [-1, 1])
    ymax = tf.reshape(ground_truth[:, 1], [-1, 1])
    xmin = tf.reshape(ground_truth[:, 2], [-1, 1])
    xmax = tf.reshape(ground_truth[:, 3], [-1, 1])
    class_id = tf.reshape(ground_truth[:, 4], [-1, 1])
    y = (ymin + ymax) / 2.
    x = (xmin + xmax) / 2.
    h = ymax - ymin
    w = xmax - xmin 
    ground_truth_= tf.concat([y, x, h, w, class_id], axis=-1)
    return ground_truth_ 

def pad_truth(ground_truth, pad_truth_to):
    ground_truth_ = tf.pad(  ground_truth, 
                            [[0, pad_truth_to -tf.shape(ground_truth)[0]], [0, 0]],
                            constant_values=-1.0
                        )
    return ground_truth_
      