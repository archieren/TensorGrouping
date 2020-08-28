import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

import tensorgroup.models.networks.layers.rsu as RSU

inputs = KL.Input(shape=(64,64,3), dtype=tf.float32)

outputs= RSU.RSU7()(inputs)

model = KM.Model(inputs=inputs,outputs=outputs)
model.summary()
