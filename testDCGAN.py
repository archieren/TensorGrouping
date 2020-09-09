import tensorflow as tf  # TF 2.0
import os

from tensorgroup.models.networks.DCGANBuilder import DCGANBuilder


K = tf.keras
KA = tf.keras.applications
KL = tf.keras.layers
KO = tf.keras.optimizers

KD = tf.keras.datasets
KM = tf.keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

builder = DCGANBuilder()

D = builder.D(name='Dis', is_sn=True)
D.summary()

E = builder.E(name='Enc')
E.summary()

G = builder.G(name='Gen')
G.summary()
