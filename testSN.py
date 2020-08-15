import tensorflow as tf  # TF 2.0
import os

from tensorgroup.models.networks.layers.sn import SpectralNormalization


KA = tf.keras.applications
KL = tf.keras.layers
KO = tf.keras.optimizers

KD = tf.keras.datasets
KM = tf.keras.models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

batch_size = 200
buffer_size = 2000
num_epochs = 6 # 200

(train_images, train_labels), (_, _) = KD.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')

train_images = train_images / 255.0
train_labels = train_labels.astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size).batch(batch_size)


inputs = KL.Input(shape=(28,28,1), dtype=tf.float32)

x = SpectralNormalization(KL.Conv2D(32, (3, 3), activation='relu'))(inputs)
x = KL.MaxPooling2D((2, 2))(x)
x = SpectralNormalization(KL.Conv2D(64, (3, 3), activation='relu'))(x)
x = KL.MaxPooling2D((2, 2))(x)
x = KL.Flatten()(x)
x = KL.Dense(64, activation='relu')(x)
output = KL.Dense(10, activation='softmax')(x)
model = KM.Model(inputs=inputs, outputs=output)

def loss(model, x, y):
    y_ = model(x, training=True)
    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
      loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss_results = []
train_accuracy_results = []

print(model.trainable_variables)

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg(loss_value)
        epoch_accuracy(y, model(x, training=False))

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print("Epoch {:03d}: Loss: {:.3f}, Acc: {:.3%}".format(epoch,
                                                           epoch_loss_avg.result(),
                                                           epoch_accuracy.result()))

# model.compile(optimizer= optimizer,loss= loss_object)
# model.fit(train_dataset)
