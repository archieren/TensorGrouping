import os
import numpy as np
from PIL import Image, ImageDraw
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
u_net_input_config = {
    'data_format': 'channels_last',
    'network_input_size': [I_SIZE, I_SIZE],                          # Must match the network's input_shape!
    'in_name': 'image',
    'out_name': 'side_fuse'                                          # 'side_all'
}


# model = UB.UnetBuilder.unet()
# model.summary()


def train(dataset='catenary'):
    tfr_dir = "./data_u_2_mask/catenary/tf_records"
    inputs_definer = DefineInputs
    trainset = mask.MaskInputs(tfr_dir, inputs_definer=inputs_definer, batch_size=2, num_exsamples=-1, repeat_num=2, buffer_size=1000)

    checkpoint_dir = os.path.join(os.getcwd(), 'work', 'u_net', dataset, 'ckpt')
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'u_net', dataset, 'sm')
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    if not os.path.exists(saved_model_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_model_dir)

    train_model = UB.UnetBuilder.unet(input_size=(I_SIZE, I_SIZE, 3))
    train_model.compile(optimizer=KO.Adam(lr=1e-3), loss='binary_crossentropy')

    checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')
    cp_callback = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        train_model.load_weights(latest)
    train_model.fit(trainset(u_net_input_config), epochs=50, callbacks=[cp_callback])
    train_model.save(os.path.join(saved_model_dir, '{}_model.h5'.format(dataset)))

# make_dataset()

def normPred(p):
    max = KB.max(p)
    min = KB.min(p)
    p = (p - min)/(max - min)
    return p

def predict(dataset='catenary'):
    saved_model_dir = os.path.join(os.getcwd(), 'work', 'u_net', dataset, 'sm')
    model = UB.UnetBuilder.unet(input_size=(I_SIZE, I_SIZE, 3))
    model.load_weights(os.path.join(saved_model_dir, '{}_model.h5'.format(dataset)), by_name=True, skip_mismatch=True)
    path = os.path.join(os.getcwd(), 'data_u_2_mask', dataset, 'TestImages', '3.jpg')
    image = Image.open(path)
    image = image.convert("RGB")
    image = np.array(image)
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


# train()
predict()
# about_mask_dataset()
