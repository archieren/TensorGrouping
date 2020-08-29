import tensorflow as tf  # TF 2.0
import tensorgroup.models.networks.U2netBuilder as U2B

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers



model = U2B.U2netBuilder.u_2_net(input_shape=(None,None,3))
model.summary()

model = U2B.U2netBuilder.u_2_net_p(input_shape=(32,32,3))
model.summary()
