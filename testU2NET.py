import tensorflow as tf  # TF 2.0
import tensorgroup.models.networks.U2netBuilder as U2B
import tensorgroup.models.networks.UnetBuilder as UB

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

model = U2B.U2netBuilder.u_2_net(input_shape=(None, None, 3))
model.summary()

model = U2B.U2netBuilder.u_2_net_p(input_shape=(32, 32, 3))
model.summary()

model = UB.UnetBuilder.unet()
model.summary()

x = KL.Input(shape=(256, 256, 3), name='image')
model = U2B.U_2_Net(side_c=3, is_simple=True)
y = model(x)
model.summary()
