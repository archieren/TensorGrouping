# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras as K

from tensorgroup.models.dataset.coco import coco_kp as KP
from tensorgroup.models.dataset import mode_keys as ModeKey

KL = tf.keras.layers
KO = tf.keras.optimizers

DATASETNAME = 'three_point'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.

lr = 0.001
batch_size = 15
buffer_size = 256
epochs = 160
reduce_lr_epoch = []

MAX_POINTS = 10

I_SIZE, I_CH = 512, 3                        # 512, 3

keypointnet_input_config = {
    'data_format': 'channels_last',
    'network_input_shape': [I_SIZE, I_SIZE],  # Must match the network's input_shape!
    'network_input_channels': I_CH,
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'color_jitter_prob': 0.5,
    'pad_truth_to': MAX_POINTS,                                   # Must match the maximal objects!
}

def about_dataset_coco_kp(datasetName='three_point'):
    from tensorgroup.models.dataset.keypointnet_inputs import DefineInputs
    from tensorgroup.models.dataset import mode_keys as MK
    tfr_dir = os.path.join(os.getcwd(), "data_coco_kp", datasetName, "tf_records")
    inputs_definer = DefineInputs
    dataset = KP.CocoKpInput(tfr_dir, inputs_definer=inputs_definer, mode=MK.TRAIN, batch_size=2, num_exsamples=2)

    print("hello -----------------------------------------------")
    for inputs, targets in dataset(keypointnet_input_config):
        image_d = inputs['image'][0]
        image = Image.fromarray(np.uint8(image_d * 255))
        image.show()
        print("\n")
        print(tf.shape(inputs['image']))
        print(tf.shape(inputs['indices_pos']))
        print(tf.shape(inputs['indices_mask']))
        print(tf.shape(inputs['center_offset']))
        # print(tf.shape(inputs['shape']))
        print(tf.shape(inputs['center_keypoint_heatmap']))
        print(tf.shape(inputs['center_keypoint_mask']))
        print(tf.shape(targets['loss_as_output']))

def about_resnetkeypoint():
    from tensorgroup.models.networks import ResnetKeypointBuilder as RKB
    train_model, prediction_model, debug_model = RKB.ResnetKeypointBuilder.build_keypoint_resnet_101(384, 3, 2)
    train_model.summary()
    prediction_model.summary()
    debug_model.summary()

def train(datasetName="three_point"):
    from tensorgroup.models.dataset.keypointnet_inputs import DefineInputs
    from tensorgroup.models.networks import ResnetKeypointBuilder as RKB

    checkpoint_dir = os.path.join(os.getcwd(), 'work', 'keypointnet', datasetName, 'ckpt')
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'keypointnet', datasetName, 'sm')
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    if not os.path.exists(saved_model_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_model_dir)

    tfr_dir = os.path.join(os.getcwd(), 'data_coco_kp', datasetName, 'tf_records')  # "./data_voc/tf_records"
    inputs_definer = DefineInputs
    dataset = KP.CocoKpInput(tfr_dir,
                             datasetName=datasetName,
                             inputs_definer=inputs_definer,
                             mode=ModeKey.TRAIN,
                             batch_size=4,
                             num_exsamples=-1,
                             repeat_num=2,
                             buffer_size=10000)
    train_model, _, _ = RKB.ResnetKeypointBuilder.build_keypoint_resnet_101(I_SIZE, I_CH, 2)  # I_CH
    # train_model.summary()

    def center_loss(y_true, y_pred):
        return y_pred

    train_model.compile(optimizer=KO.Adam(lr=1e-4), loss={'loss_as_output': center_loss})

    checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
    cp_callback = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        train_model.load_weights(latest)
    train_model.fit(dataset(keypointnet_input_config), epochs=300, callbacks=[cp_callback])
    train_model.save(os.path.join(saved_model_dir, '{}.h5'.format(datasetName)))

def predict(datasetName='three_keypoint'):
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'keypointnet', datasetName, 'sm')
    images_dir = os.path.join(os.getcwd(), 'data_coco_kp', datasetName, 'TestImages')

    from tensorgroup.models.networks import ResnetKeypointBuilder as RKB
    from tensorgroup.models.dataset.coco import coco_kp as CKP

    _, predict_model, _ = RKB.ResnetKeypointBuilder.build_keypoint_resnet_101(I_SIZE, I_CH, 2)

    def load_image(images_dir, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(images_dir, '({}).jpg'.format(image_index))
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(path)
        image = image.convert("RGB")
        return np.array(image), image

    predict_model.load_weights(os.path.join(saved_model_dir, '{}.h5'.format(datasetName)), by_name=True, skip_mismatch=True)
    for index in range(1, 6):
        image_array, image = load_image(images_dir, index)
        draw = ImageDraw.Draw(image)
        print(image.size)
        print(image_array.shape)
        h, w = image.size  # 作了假设的哈：image.shape[2]=I_CH，偷懒。Bad smell

        image_t = tf.convert_to_tensor(image_array)
        image_t = CKP.CocoKpInput.ImageNormalizer()(image_t)
        image_t = tf.image.resize(image_t, keypointnet_input_config['network_input_shape'], method=tf.image.ResizeMethod.BILINEAR)
        image_input = tf.expand_dims(image_t, axis=0)
        predicts = predict_model.predict(image_input)[0]
        scores = predicts[:, 2]
        indices = np.where(scores > 0.10)
        detections = predicts[indices].copy()
        scale_w = (I_SIZE / w) * 0.25  # 注意
        scale_h = (I_SIZE / h) * 0.25
        for detection in detections:
            cx = int(round(detection[0])/scale_w)
            cy = int(round(detection[1])/scale_h)
            score = '{:.4f}'.format(detection[2])
            # class_id = int(detection[3])
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 6)
            print("{} {} {}".format(cx, cy, score))
            draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill=(255, 0, 0), outline=(255, 0, 0), width=1)
        plt.imshow(image)
        plt.show()

def build_Data_From_JSONs():  # 有必要注释一下目录结构 和约定目录结构 TODO
    root = os.path.join(os.getcwd(), 'data_coco_kp', DATASETNAME)
    source_dir = os.path.join(root, 'data', ModeKey.TRAIN)
    output_dir = os.path.join(root, 'tf_records')
    if not os.path.exists(output_dir):   #
        os.makedirs(output_dir)
    KP.produce_dataset_from_jsons(dataset_name=ModeKey.TRAIN, json_source_dir=source_dir, target_directory=output_dir)
    pass


if __name__ == '__main__':
    # build_Data_From_JSONs()
    # about_dataset_coco_kp()
    # about_resnetkeypoint()
    # train(datasetName="three_point")
    predict(datasetName="three_point")
