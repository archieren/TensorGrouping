import os
# import hashlib

# import multiprocessing
# import numpy as np
# import skimage.io as io

from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from lxml import etree
from tensorgroup.models.dataset import mode_keys as ModeKey

KA = tf.keras.applications
KL = tf.keras.layers
KO = tf.keras.optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


image_shape = (384, 384, 3)

lr = 0.001
batch_size = 15
buffer_size = 256
epochs = 160
reduce_lr_epoch = []

MAX_OBJECTS = 100

config = {
    'mode': 'train',                                       # 'train', 'test'
    'input_size': 384,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.1,
    'top_k_results_output': MAX_OBJECTS,

}

centernet_input_config = {
    'data_format': 'channels_last',
    'network_input_shape': [512, 512],                           # Must match the network's input_shape!
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'color_jitter_prob': 0.5,
    'pad_truth_to': MAX_OBJECTS,                                   # Must match the maximal objects!
}

def about_dataset_voc():
    from tensorgroup.models.dataset.centernet_inputs import DefineInputs
    from tensorgroup.models.dataset.voc import voc
    from tensorgroup.models.dataset import mode_keys as MK
    inputs_definer = DefineInputs
    dataset = voc.VocInput(inputs_definer=inputs_definer, mode=MK.TRAIN, batch_size=2, num_exsamples=4)

    for image, ground_truth, center_round, center_offset, shape_offset, center_keypoint_heatmap, center_keypoint_mask in dataset(centernet_input_config):
        # plt.imshow(image[1])
        # plt.show()
        print(tf.shape(image))
        print(tf.shape(center_keypoint_heatmap))
        print(tf.shape(center_keypoint_mask))
        print(tf.shape(ground_truth))
        print(tf.shape(center_round))
        print("\n")

def make_voc_custom_dataset():
    from tensorgroup.models.dataset.voc import voc_custom
    ann_dir, img_dir, tfr_dir = "./data_voc/Annotations", "./data_voc/Annotations", "./data_voc/tf_records"
    voc_custom.dataset2tfrecord(ann_dir, img_dir, tfr_dir, ModeKey.TRAIN)

def about_dataset_voc_custom():
    from tensorgroup.models.dataset.centernet_inputs import DefineInputs
    from tensorgroup.models.dataset.voc import voc_custom
    tfr_dir = "./data_voc/tf_records"
    inputs_definer = DefineInputs
    dataset = voc_custom.VocCustomInput(tfr_dir, inputs_definer=inputs_definer, batch_size=2, num_exsamples=200, repeat_num=1, buffer_size=10000)

    for image, indices, indices_mask, center_offset, shape_offset, center_keypoint_heatmap, center_keypoint_mask in dataset(centernet_input_config):
        # plt.imshow(image[1])
        # plt.show()
        print(tf.shape(center_keypoint_heatmap))
        print(tf.shape(center_keypoint_mask))
        print(tf.shape(indices))
        print(tf.shape(indices_mask))
        print("\n")

def repair_data(ann_dir):
    xmllist = tf.io.gfile.glob(os.path.join(ann_dir, '*.xml'))
    for xmlpath in xmllist:
        xml = etree.parse(xmlpath)
        root = xml.getroot()
        image_name = root.find('filename')
        path = root.find('path')
        path.text = image_name.text
        xml.write(xmlpath)

def test_gather():
    # 琢磨一下 tf.gather_nd
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    print("a shape is {}".format(a.shape))
    # b = np.array([[[0, 1]], [[2, 3]]])
    b = np.array([[0], [0], [3], [3]])
    print("b shape is {}".format(b.shape))
    print(tf.gather_nd(a, b, batch_dims=1))
    print(tf.gather_nd(a, b, batch_dims=0))
    print("-------------------------------")
    b = np.array([[0, 1], [2, 3]])
    print("b shape is {}".format(b.shape))
    # print(tf.gather_nd(a, b, batch_dims=1))
    print(tf.gather_nd(a, b, batch_dims=0))

def train():
    from tensorgroup.models.dataset.centernet_inputs import DefineInputs
    from tensorgroup.models.dataset.voc import voc_custom
    from tensorgroup.models.networks import CenterNetBuilder as CNB
    tfr_dir = "./data_voc/tf_records"
    inputs_definer = DefineInputs
    dataset = voc_custom.VocCustomInput(tfr_dir, inputs_definer=inputs_definer, batch_size=2, num_exsamples=200, repeat_num=1, buffer_size=10000)
    train_model, _, _ = CNB.CenterNetBuilder.CenterNetOnResNet50V2(len(voc_custom.voc_custom_classes))
    train_model.summary()

    def center_loss(y_true, y_pred):
        return y_pred

    train_model.compile(optimizer=KO.Adam(lr=1e-3), loss={'centernet_loss': center_loss})
    # train_model.fit(dataset)


if __name__ == '__main__':
    # about_dataset_voc()
    # repair_data("./data_voc/Annotations/")
    # tf.executing_eagerly()
    # make_voc_custom_dataset()
    # about_dataset_voc_custom()
    # test_gather()
    # test_meshgrid()
    # test_gengaussian()
    train()
