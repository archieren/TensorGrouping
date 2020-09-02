import tensorflow as tf  # TF 2.0
import os
import datetime

from tensorgroup.models.networks.layers.sn import SpectralNormalization
from tensorgroup.models.networks.layers.sa import Attention, SN_Attention

K  = tf.keras
KA = tf.keras.applications
KL = tf.keras.layers
KO = tf.keras.optimizers

KD = tf.keras.datasets
KM = tf.keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

batch_size = 200
buffer_size = 2000
num_epochs = 20 # 200

(train_images, train_labels), (_, _) = KD.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')

train_images = train_images / 255.0
train_labels = train_labels.astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size).batch(batch_size)


inputs = KL.Input(shape=(28,28,1), dtype=tf.float32)

x = SpectralNormalization(KL.Conv2D(32, (3, 3), activation='relu'))(inputs)
x = KL.MaxPooling2D((2, 2))(x)
x = SN_Attention(32)(x)
x = SpectralNormalization(KL.Conv2D(64, (3, 3), activation='relu'))(x)
x = KL.MaxPooling2D((2, 2))(x)
x = KL.Flatten()(x)
x = KL.Dense(64, activation='relu')(x)
output = KL.Dense(10, activation='softmax')(x)
model = KM.Model(inputs=inputs, outputs=output)
model.summary()
def loss(model, x, y):
    y_ = model(x, training=True)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

checkpoint_dir = os.path.join(os.getcwd(), 'work', 'testSN', 'ckpt')
saved_model_dir = os.path.join(os.getcwd(), 'work', 'testSN', 'sm')

if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
    os.makedirs(checkpoint_dir)
if not os.path.exists(saved_model_dir):   # model_dir 不应出现这种情况.
    os.makedirs(saved_model_dir)

checkpoint_path = os.path.join(checkpoint_dir, 'cp.ckpt')


latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is not None:
    model.load_weights(latest)


####################################################################################By Fit
"""
log_dir=os.path.join(os.getcwd(), 'work','testSN', 'log', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(log_dir):  os.makedirs(log_dir)
cp_callback = K.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.compile(optimizer= optimizer,loss= loss_object)
model.fit(train_dataset, epochs=num_epochs, callbacks=[cp_callback, tensorboard_callback])
model.save(os.path.join(saved_model_dir, 'testSN.h5'))
"""
####################################################################################
train_log_dir =os.path.join(os.getcwd(), 'work','testSN', 'log', 'gradient_tape', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
#test_log_dir =os.path.join(os.getcwd(), 'work','testSN', 'log', 'gradient_tape', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'test')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#test_summary_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(num_epochs):
    train_epoch_loss = tf.keras.metrics.Mean()
    train_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_epoch_loss(loss_value)
        train_epoch_accuracy(y, model(x, training=False))

    print("Epoch {:03d}: Loss: {:.3f}, Acc: {:.3%}".format(epoch,
                                                           train_epoch_loss.result(),
                                                           train_epoch_accuracy.result()))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_epoch_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_epoch_accuracy.result(), step=epoch)

    train_epoch_loss.reset_states()
    train_epoch_accuracy.reset_states()

