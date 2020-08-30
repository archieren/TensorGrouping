import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers


def down_with(down_in, filters, droping=False,):
    hor_out = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down_in)
    hor_out = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(hor_out)
    if droping:
        hor_out = KL.Dropout(0.5)(hor_out)
    down_out = KL.MaxPooling2D(pool_size=(2, 2))(hor_out)
    return hor_out, down_out

def floor(down_in, filters):
    conv = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(down_in)
    conv = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    up_in = KL.Dropout(0.5)(conv)
    return up_in

def up_with(up_in, hor_in, filters):
    up = KL.Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal')(KL.UpSampling2D(size=(2, 2))(up_in))
    merge = KB.concatenate([hor_in, up])
    up_out = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    up_out = KL.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up_out)
    return up_out

class UnetBuilder(object):
    @staticmethod
    def unet(input_size=(256, 256, 1)):
        inputs = KL.Input(input_size)
        down = inputs
        # Down
        hor_1, down = down_with(down, 64)
        hor_2, down = down_with(down, 128)
        hor_3, down = down_with(down, 256)
        hor_4, down = down_with(down, 512, droping=True)

        # Floor
        up = floor(down, 1024)
        # Up
        up = up_with(up, hor_4, 512)
        up = up_with(up, hor_3, 256)
        up = up_with(up, hor_2, 128)
        up = up_with(up, hor_1, 64)

        outputs = KL.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)
        outputs = KL.Conv2D(1, 1, activation='sigmoid')(outputs)

        model = KM.Model(inputs=inputs, outputs=outputs)

        #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        #model.summary()
        return model
