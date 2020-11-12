import os
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf  # TF 2.0
import tensorflow.keras as K
import tensorgroup.models.networks.U2netBuilder as U2B
import tensorgroup.models.networks.UnetBuilder as UB
from tensorgroup.models.dataset.u_net_inputs import DefineInputs
from tensorgroup.models.dataset.unet_mask import mask
from tensorgroup.models.dataset import mode_keys as ModeKey

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers
KO = tf.keras.optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.


I_SIZE = 512
u_2_net_input_config = {
    'data_format': 'channels_last',
    'network_input_size': [I_SIZE, I_SIZE],                          # Must match the network's input_shape!
    'in_name': 'image',
    'out_name': 'side_fuse'                                          # 'side_all'
}

# model = U2B.U2netBuilder.u_2_net(input_shape=(1024, 1024, 3))
# model.summary()

# model = U2B.U2netBuilder.u_2_net_p(input_shape=(1024, 1024, 3))
# model.summary()

# model = UB.UnetBuilder.unet()
# model.summary()

# x = KL.Input(shape=(256, 256, 3), name='image')
# model = U2B.U_2_Net(side_c=3, is_simple=True)
# y = model(x)
# model.summary()

def make_dataset():
    root = './data_u_2_mask/catenary/'
    img_dir = os.path.join(root, 'JPEGImages')
    mask_ann_dir = os.path.join(root, 'Annotations')
    output_dir = os.path.join(root, 'tf_records')
    mask.dataset2tfrecord(img_dir, mask_ann_dir, output_dir, ModeKey.TRAIN)
    return

def about_mask_dataset():
    tfr_dir = "./data_u_2_mask/catenary/tf_records"
    inputs_definer = DefineInputs
    dataset = mask.MaskInputs(tfr_dir, inputs_definer=inputs_definer, batch_size=2, num_exsamples=200, repeat_num=1, buffer_size=1000)

    for inputs, targets in dataset(u_2_net_input_config):
        print(tf.shape(inputs[u_2_net_input_config['in_name']]))
        print(tf.shape(targets[u_2_net_input_config['out_name']]))

# about_mask_dataset()

def train():
    tfr_dir = "./data_u_2_mask/catenary/tf_records"
    inputs_definer = DefineInputs
    dataset = mask.MaskInputs(tfr_dir, inputs_definer=inputs_definer, batch_size=2, num_exsamples=-1, repeat_num=2, buffer_size=1000)

    checkpoint_dir = os.path.join(os.getcwd(), 'work', 'u_2_net_p', 'ckpt')
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'u_2_net_p', 'sm')
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    if not os.path.exists(saved_model_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_model_dir)

    train_model = U2B.U2netBuilder.u_2_net_p(input_shape=(I_SIZE, I_SIZE, 3))
    train_model.compile(optimizer=KO.Adam(lr=1e-3), loss='binary_crossentropy')

    checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
    cp_callback = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        train_model.load_weights(latest)
    train_model.fit(dataset(u_2_net_input_config), epochs=200, callbacks=[cp_callback])
    train_model.save(os.path.join(saved_model_dir, 'catenary_model.h5'))


def normPred(p):
    max = KB.max(p)
    min = KB.min(p)
    p = (p - min)/(max - min)
    return p

def predict():
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'u_2_net_p', 'sm')
    model = U2B.U2netBuilder.u_2_net_p(input_shape=(I_SIZE, I_SIZE, 3))
    model.load_weights(os.path.join(saved_model_dir, 'catenary_model.h5'), by_name=True, skip_mismatch=True)
    path = os.path.join(os.getcwd(), 'data_u_2_mask', 'catenary', 'TestImages', '9.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_t = tf.convert_to_tensor(image)
    image_t = mask.MaskInputs.ImageNormalizer()(image_t)
    print(tf.reduce_max(image_t))
    image_t = tf.image.resize(image_t, [1024, 1024], method=tf.image.ResizeMethod.BILINEAR)
    image_input = tf.expand_dims(image_t, axis=0)
    predict = model.predict(image_input)[0]
    # predict = normPred(predict)
    plt.imshow(predict)
    plt.show()
    pass


# make_dataset()
# train()
predict()
# about_mask_dataset()

def about_resnet_fpn():
    import tensorgroup.models.networks.ResnetWithFPN as FPN
    image_input = KL.Input(shape=(1024, 1024, 3), name='image')
    fpn = FPN.ResnetBuilder.build_resnet_50_fpn(image_input, 10)
    model = KM.Model(inputs=image_input, outputs=fpn)
    model.summary()


# about_resnet_fpn()
