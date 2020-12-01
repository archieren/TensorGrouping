import os
# import hashlib

# import multiprocessing
# import numpy as np
# import skimage.io as io

from matplotlib import pyplot as plt

import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
import tensorflow.keras as K
from lxml import etree
from tensorgroup.models.dataset import mode_keys as ModeKey

KA = tf.keras.applications
KL = tf.keras.layers
KO = tf.keras.optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.


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

I_SIZE, I_CH = 1024, 3                        # 512, 3

centernet_input_config = {
    'data_format': 'channels_last',
    'network_input_shape': [I_SIZE, I_SIZE],  # Must match the network's input_shape!
    'network_input_channels': I_CH,
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

    for inputs, targets in dataset(centernet_input_config):
        plt.imshow(inputs['image'][0])
        plt.show()
        print(inputs['indices_pos'])
        print("\n")

def make_voc_custom_dataset(datasetName='lanzhou'):
    from tensorgroup.models.dataset.voc import voc_custom
    prefix = os.path.join(os.getcwd(), 'data_voc', datasetName)
    ann_dir = os.path.join(prefix, "Annotations")
    img_dir = os.path.join(prefix, "Annotations")
    tfr_dir = os.path.join(prefix, "tf_records")
    voc_custom.dataset2tfrecord(ann_dir, img_dir, tfr_dir, ModeKey.TRAIN, datasetName=datasetName)

def about_dataset_voc_custom(datasetName='lanzhou'):
    from tensorgroup.models.dataset.centernet_inputs import DefineInputs
    from tensorgroup.models.dataset.voc import voc_custom
    tfr_dir = os.path.join(os.getcwd(), "data_voc", datasetName, "tf_records")
    inputs_definer = DefineInputs
    dataset = voc_custom.VocCustomInput(tfr_dir, datasetName=datasetName, inputs_definer=inputs_definer, batch_size=2, num_exsamples=200, repeat_num=1, buffer_size=10000)
    # dataset 来的输入如下:（参见centernet_inputs.py）
    # {'image': image,
    #  'indices': indices,
    #  'indices_mask': indices_mask,
    #  'center_offset': center_offset,
    #  'shape': shape,
    #  'center_keypoint_heatmap': center_keypoint_heatmap,
    #  'center_keypoint_mask': center_keypoint_mask}
    for inputs, targets in dataset(centernet_input_config):
        print(tf.shape(inputs['image']))
        print(tf.shape(inputs['indices_pos']))
        print(tf.shape(inputs['indices_mask']))
        print(tf.shape(inputs['center_offset']))
        print(tf.shape(inputs['shape']))
        print(tf.shape(inputs['center_keypoint_heatmap']))
        print(tf.shape(inputs['center_keypoint_mask']))
        print(tf.shape(targets['loss_as_output']))

def repair_data(ann_dir):
    xmllist = tf.io.gfile.glob(os.path.join(ann_dir, '*.xml'))
    for xmlpath in xmllist:
        xml = etree.parse(xmlpath)
        root = xml.getroot()
        filename = root.find('filename')
        path = root.find('path')
        path.text = filename.text
        depth = root.find('size').find('depth')
        depth.text = '3'
        # xml.write(xmlpath)
        for child in root:
            if child.tag == 'object':
                name = child.find('name')
                if name.text.find('clamp') > -1 or name.text.find('wire') > -1:
                    name.text = 'holder'
                    print(name.text)
        xml.write(xmlpath)

def train(datasetName="lanzhou"):
    from tensorgroup.models.dataset.centernet_inputs import DefineInputs
    from tensorgroup.models.dataset.voc import voc_custom
    from tensorgroup.models.networks import CenterNetBuilder as CNB

    checkpoint_dir = os.path.join(os.getcwd(), 'work', 'centernet', datasetName, 'ckpt')
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'centernet', datasetName, 'sm')
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    if not os.path.exists(saved_model_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_model_dir)

    tfr_dir = os.path.join(os.getcwd(), 'data_voc', datasetName, 'tf_records')  # "./data_voc/tf_records"
    inputs_definer = DefineInputs
    dataset = voc_custom.VocCustomInput(tfr_dir, datasetName=datasetName, inputs_definer=inputs_definer, batch_size=2, num_exsamples=-1, repeat_num=2, buffer_size=10000)
    train_model, _, _ = CNB.CenterNetBuilder.CenterNetOnResNet50V2(len(voc_custom.voc_custom_classes[datasetName]),
                                                                   input_size=I_SIZE,
                                                                   input_channels=I_CH)  # I_CH
    # train_model.summary()

    def center_loss(y_true, y_pred):
        return y_pred

    train_model.compile(optimizer=KO.Adam(lr=1e-4), loss={'loss_as_output': center_loss})

    checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
    cp_callback = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        train_model.load_weights(latest)
    train_model.fit(dataset(centernet_input_config), epochs=888, callbacks=[cp_callback])
    train_model.save(os.path.join(saved_model_dir, '{}.h5'.format(datasetName)))

def load_image(images_dir, image_index):
    """
    Load an image at the image_index.
    """
    path = os.path.join(images_dir, '({}).jpg'.format(image_index))
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.open(path)
    image = image.convert("RGB")
    return np.array(image)

def save_image(images_dir, image, i, j):
    path = os.path.join(images_dir, '({}-{}).jpg').format(i, j)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(path, image)
    image = Image.fromarray(image)
    image.save(path)
    return

def predict(datasetName='lanzhou'):
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'centernet', datasetName, 'sm')
    images_dir = os.path.join(os.getcwd(), 'data_voc', datasetName, 'TestImages')

    from tensorgroup.models.networks import CenterNetBuilder as CNB
    from tensorgroup.models.dataset.voc import voc_custom

    _, predict_model, _ = CNB.CenterNetBuilder.CenterNetOnResNet50V2(len(voc_custom.voc_custom_classes[datasetName]),
                                                                     input_size=I_SIZE,
                                                                     input_channels=I_CH,
                                                                     score_threshold=0.01)

    predict_model.load_weights(os.path.join(saved_model_dir, '{}.h5'.format(datasetName)), by_name=True, skip_mismatch=True)
    for index in range(262, 266):
        image_array = load_image(images_dir, index)
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)
        image_size = image_array.shape[0]   # 作了假设的哈：image.shape[2]=I_CH，偷懒。Bad smell

        image_t = tf.convert_to_tensor(image_array)
        image_t = voc_custom.VocCustomInput.ImageNormalizer()(image_t)
        image_t = tf.image.resize(image_t, centernet_input_config['network_input_shape'], method=tf.image.ResizeMethod.BILINEAR)
        image_input = tf.expand_dims(image_t, axis=0)
        predicts = predict_model.predict(image_input)[0]
        scores = predicts[:, 4]
        indices = np.where(scores > 0.1)
        detections = predicts[indices].copy()
        print(detections.shape)
        scale = (I_SIZE / image_size) * 0.25  # 注意
        for detection in detections:
            xmin = int(round(detection[0])/scale)
            ymin = int(round(detection[1])/scale)
            xmax = int(round(detection[2])/scale)
            ymax = int(round(detection[3])/scale)
            # score = '{:.4f}'.format(detection[4])
            # class_id = int(detection[5])
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 6)
            draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=(255, 0, 0), width=6)
        plt.imshow(image)
        plt.show()


def save_detection(datasetName='lanzhou'):
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'centernet', datasetName, 'sm')
    images_dir = os.path.join(os.getcwd(), 'data_voc', datasetName, 'TestImages')
    d_patch_dir = os.path.join(os.getcwd(), 'work', 'centernet', datasetName, 'detected_paches')
    if not os.path.exists(d_patch_dir):   # model_dir 不应出现这种情况.
        os.makedirs(d_patch_dir)

    from tensorgroup.models.networks import CenterNetBuilder as CNB
    from tensorgroup.models.dataset.voc import voc_custom

    _, predict_model, _ = CNB.CenterNetBuilder.CenterNetOnResNet50V2(len(voc_custom.voc_custom_classes[datasetName]),
                                                                     input_size=I_SIZE,
                                                                     input_channels=I_CH,
                                                                     score_threshold=0.01)

    predict_model.load_weights(os.path.join(saved_model_dir, '{}.h5'.format(datasetName)), by_name=True, skip_mismatch=True)
    for index in range(254, 266):
        image_array = load_image(images_dir, index)
        image_size = image_array.shape[0]   # 作了假设的哈, 偷懒。Bad smell

        image_t = tf.convert_to_tensor(image_array)
        image_t = voc_custom.VocCustomInput.ImageNormalizer()(image_t)
        image_t = tf.image.resize(image_t, centernet_input_config['network_input_shape'], method=tf.image.ResizeMethod.BILINEAR)
        image_input = tf.expand_dims(image_t, axis=0)
        predicts = predict_model.predict(image_input)[0]
        scores = predicts[:, 4]
        indices = np.where(scores > 0.1)
        detections = predicts[indices].copy()
        print(detections.shape)
        scale = (I_SIZE / image_size) * 0.25  # 注意
        d_patches = []
        for detection in detections:
            xmin = int(round(detection[0])/scale)
            ymin = int(round(detection[1])/scale)
            xmax = int(round(detection[2])/scale)
            ymax = int(round(detection[3])/scale)
            d_patch = image_array[ymin:ymax+1, xmin:xmax+1, :].copy()
            d_patches.append(d_patch)
        for j in range(len(d_patches)):
            patch_size = d_patches[j].shape
            if patch_size[0] > 31 and patch_size[1] > 31:
                save_image(d_patch_dir, d_patches[j], index, j)


if __name__ == '__main__':
    # about_dataset_voc()
    # repair_data("./data_voc/catenary/Annotations/")
    # tf.executing_eagerly()
    # make_voc_custom_dataset(datasetName='catenary')
    # about_dataset_voc_custom(datasetName='catenary')
    # train('catenary')
    predict(datasetName='catenary')
    # save_detection(datasetName='catenary')
