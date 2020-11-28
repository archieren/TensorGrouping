import tensorflow as tf
import numpy as np

from tensorgroup.models.dataset.image_augmentor import image_augmentor

STRIDE = 4  # CenterNet的Input Shape 和 Feature Map Shape的比

class DefineInputs:
    """
    """

    def __init__(self, config, num_classes, max_objects):
        """ 指定网络需要的数据输入格式。
        Args:
            config:
              {'data_format': 'channels_last',
               'network_input_shape': [512, 512],                           # Must match the network's input_shape!
               'network_input_channels': 3
               'flip_prob': [0., 0.5],
               'fill_mode': 'BILINEAR',
               'color_jitter_prob': 0.5,
               'pad_truth_to': 100,                                   # Must match the maximal objects!
              }
            num_classes:
        """
        self._config = config
        self._num_classes = num_classes
        self._max_objects = max_objects

    def __call__(self, image, ground_truth):
        """将features转换成模型需要的形式
        Args:
            image: Normalized image.
            ground_truth: Unpadded Ground truth! 将补充为。。。
        Returns: 将loss里需要的输入全部事先计算出来
            image: Resized image.
            ground_truth：Padded Groundtruth.
            center_round, center_offset, shape_offset, center_keypoint_heatmap, center_keypoint_mask:看网络的介绍.

        """
        image, ground_truth = image_augmentor(image=image,
                                              ground_truth=ground_truth,
                                              **self._config
                                              )
        # ground_truth: [y_center, x_center, height, width, classid]
        indices, indices_mask, center_offset, shape, center_keypoint_heatmap, center_keypoint_mask = self._def_inputs(image, ground_truth)
        # 在这个地方，有必要给些注释：Model.fit 对 DataSet输入格式，是有要求的！
        return ({'image': image,
                 'indices_pos': indices,
                 'indices_mask': indices_mask,
                 'center_offset': center_offset,
                 'shape': shape,
                 'center_keypoint_heatmap': center_keypoint_heatmap,
                 'center_keypoint_mask': center_keypoint_mask},
                {'loss_as_output': tf.constant([1.0])})

    def _def_inputs(self, image, ground_truth):
        """生成网络所需要的输入。
        Args:
            image: 已经调整成了self._config['network_input_shape']
            ground_truth: y_x_h_w_class格式，且归一化了, with_padding。
        Results:
               center_round: [max_objects,2]
               center_offset: [max_objects,2]
               shape_offset:  [max_objects,2]
               center_keypoint_heatmap: [num_classes, feature_map_height, feature_map_width]
               center_keypoint_mask: [num_classes, feature_map_height, feature_map_width]
        """
        center_keypoint_heatmap, center_keypoint_mask = self._gen_center_keypoint_heatmap(ground_truth)
        indices, indices_mask, center_offset, shape = self._gen_center_round_and_center_offset_and_shape_offset(ground_truth)
        return indices, indices_mask, center_offset, shape, center_keypoint_heatmap, center_keypoint_mask

    def _gen_center_keypoint_heatmap(self, ground_truth):
        network_input_shape = self._config['network_input_shape']
        (i_h, i_w) = network_input_shape
        (f_h, f_w) = (int(i_h/STRIDE), int(i_w/STRIDE))  # 注意这个STRIDE的来源

        objects_num = tf.argmin(ground_truth, axis=0)
        objects_num = objects_num[0]
        ground_truth = tf.gather(ground_truth, tf.range(0, objects_num, dtype=tf.int64))

        # ground_truth的shape应为(objects_num, 5)
        c_y = ground_truth[..., 0] * f_h
        c_x = ground_truth[..., 1] * f_w
        c_h = ground_truth[..., 2] * f_h
        c_w = ground_truth[..., 3] * f_w
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)

        center = tf.concat([tf.expand_dims(c_y, axis=-1), tf.expand_dims(c_x, axis=-1)], axis=-1)
        c_y = tf.floor(c_y)
        c_x = tf.floor(c_x)
        center_round = tf.floor(center)
        center_round = tf.cast(center_round, dtype=tf.int64)  # tf.int64 是必需的！

        center_keypoint_heatmap = gaussian2D_tf_at_any_point(objects_num, c_y, c_x, c_h, c_w, f_h, f_w)
        zero_like_heatmap = tf.expand_dims(tf.zeros([f_h, f_w], dtype=tf.float32), axis=-1)
        all_class_heatmap = []
        all_class_mask = []
        for i in range(self._num_classes):
            is_class_i = tf.equal(class_id, i)

            class_i_heatmap = tf.boolean_mask(center_keypoint_heatmap, is_class_i, axis=0)
            class_i_heatmap = tf.cond(
                tf.equal(tf.shape(class_i_heatmap)[0], 0),
                lambda: zero_like_heatmap,
                lambda: tf.expand_dims(tf.reduce_max(class_i_heatmap, axis=0), axis=-1)
            )
            all_class_heatmap.append(class_i_heatmap)

            class_i_center = tf.boolean_mask(center_round, is_class_i, axis=0)
            class_i_mask = tf.cond(
                tf.equal(tf.shape(class_i_center)[0], 0),
                lambda: zero_like_heatmap,
                lambda: tf.expand_dims(tf.sparse.to_dense(tf.sparse.SparseTensor(class_i_center, tf.ones_like(class_i_center[..., 0], tf.float32), dense_shape=[f_h, f_w]), validate_indices=False), axis=-1)
            )
            all_class_mask.append(class_i_mask)
        center_keypoint_heatmap = tf.concat(all_class_heatmap, axis=-1)
        center_keypoint_mask = tf.concat(all_class_mask, axis=-1)

        return center_keypoint_heatmap, center_keypoint_mask

    def _gen_center_round_and_center_offset_and_shape_offset(self, ground_truth):
        network_input_shape = self._config['network_input_shape']
        (i_h, i_w) = network_input_shape
        (f_h, f_w) = (int(i_h/STRIDE), int(i_w/STRIDE))
        c_y = ground_truth[..., 0] * f_h
        c_x = ground_truth[..., 1] * f_w
        c_h = ground_truth[..., 2] * f_h
        c_w = ground_truth[..., 3] * f_w

        center = tf.concat([tf.expand_dims(c_y, axis=-1), tf.expand_dims(c_x, axis=-1)], axis=-1)
        center_round = tf.floor(center)
        center_offset = center - center_round
        # center_round = tf.cast(center_round, dtype=tf.int64)  # tf.int64 是必需的！
        # indices = center_round

        shape = tf.concat([tf.expand_dims(c_h, axis=-1), tf.expand_dims(c_w, axis=-1)], axis=-1)
        # shape_round = tf.floor(shape)
        # shape_offset = shape - shape_round

        indices_mask = tf.cast(tf.greater(tf.expand_dims(c_h, axis=-1), 0.0), dtype=tf.float32) * tf.cast(tf.greater(tf.expand_dims(c_w, axis=-1), 0.0), dtype=tf.float32)
        t_mask = tf.tile(indices_mask, (1, 2))
        indices = center_round * t_mask
        indices = tf.cast(indices, dtype=tf.int64)

        return indices, indices_mask, center_offset, shape

def gaussian2D_tf_at_any_point(c_num, c_y, c_x, c_h, c_w, f_h, f_w):
    sigma = gaussian_radius_tf(c_h, c_w)
    c_y = tf.reshape(c_y, [-1, 1, 1])
    c_x = tf.reshape(c_x, [-1, 1, 1])

    y_range = tf.range(0, f_h, dtype=tf.float32)
    x_range = tf.range(0, f_w, dtype=tf.float32)
    [mesh_x, mesh_y] = tf.meshgrid(x_range, y_range)
    mesh_x = tf.expand_dims(mesh_x, 0)
    mesh_x = tf.tile(mesh_x, [c_num, 1, 1])
    mesh_y = tf.expand_dims(mesh_y, 0)
    mesh_y = tf.tile(mesh_y, [c_num, 1, 1])
    center_keypoint_heatmap = tf.exp(-((c_y-mesh_y)**2+(c_x-mesh_x)**2)/(2*sigma**2))
    return center_keypoint_heatmap


# 下面可以看到如何为网格赋予坐标的技巧！
# np.ogrid 和 tf.meshgrid的作用可以琢磨一下.
def gaussian2D(shape, sigma=1):
    # m, n = [(ss - 1.) / 2. for ss in shape]
    m, n = (shape[0]-1.)/2, (shape[1]-1.)/2
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D_tf(shape, sigma=1):
    m, n = shape[0], shape[1]
    m = tf.cast((m-1.)/2, dtype=tf.float32)
    n = tf.cast((n-1.)/2, dtype=tf.float32)

    y = tf.range(-m, m+1, dtype=tf.float32)
    x = tf.range(-n, n+1, dtype=tf.float32)
    [n_x, m_y] = tf.meshgrid(x, y)

    h = tf.exp(-(n_x ** 2 + m_y ** 2) / (2 * sigma ** 2))
    return h

def gaussian_radius_tf(height, width, min_overlap=0.7):
    """
    Args:
        height, width: Both are the tensor of the same shape (N,)!
    Results:
        radius: 考虑所有框的大小，而得到的最佳半径
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = tf.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = tf.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = tf.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return tf.reduce_min([r1, r2, r3])

def gaussian_radius(height, width, min_overlap=0.7):
    """
    Args:
        height, width: Both are the array of the same shape (N,)!
    """

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return np.min([r1, r2, r3])

def draw_gaussian(heatmap, layer, center, sigma):
    # 目标热图只采用 Gaussian类型
    (HEAT_MAP_HEIGHT, HEAT_MAP_WIDTH) = heatmap.shape[0:2]
    temperature_size = sigma * 3
    TOP, BOTTOM = LEFT, RIGHT = Y_, X_ = 0, 1  # 纯粹为了可读
    mu_y = int(center[0]+0.5)  # 调整到HeatMap的坐标系，四舍五入.
    mu_x = int(center[1]+0.5)
    # 检查Gaussian_bounds是否落在HEATMAP之外,直接跳出运行,不支持不可见JOINT POINT
    left_top = [int(mu_y - temperature_size), int(mu_x - temperature_size)]
    right_bottom = [int(mu_y + temperature_size), int(mu_x + temperature_size)]
    if left_top[Y_] >= HEAT_MAP_HEIGHT or left_top[X_] >= HEAT_MAP_WIDTH or right_bottom[Y_] < 0 or right_bottom[X_] < 0:
        assert False

    # 生成Gaussian_Area
    size = temperature_size*2+1
    g = gaussian2D([size, size], sigma)
    # 确定可用Gaussian_Area
    g_y = max(0, -left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_HEIGHT) - left_top[Y_]
    g_x = max(0, -left_top[X_]), min(right_bottom[X_], HEAT_MAP_WIDTH) - left_top[X_]
    #
    heatmap_y = max(0, left_top[Y_]), min(right_bottom[Y_], HEAT_MAP_HEIGHT)
    heatmap_x = max(0, left_top[X_]), min(right_bottom[X_], HEAT_MAP_WIDTH)

    heatmap[heatmap_y[TOP]:heatmap_y[BOTTOM], heatmap_x[LEFT]:heatmap_x[RIGHT], layer] = g[g_y[TOP]:g_y[BOTTOM], g_x[LEFT]:g_x[RIGHT]]
    pass
