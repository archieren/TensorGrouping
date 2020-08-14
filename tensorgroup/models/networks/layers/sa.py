import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

class Attention(KL.Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        # print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h, initializer='glorot_uniform', name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,), initializer='zeros', name='bias_h')
        super(Attention, self).build(input_shape)  # 这是必须的。
        # Set input spec.
        # self.input_spec = InputSpec(ndim=4,axes={3: input_shape[-1]})
        self.built = True  # 这是必须的

    def call(self, x):
        def hw_flatten(x):
            return KB.reshape(x, shape=[KB.shape(x)[0], KB.shape(x)[1]*KB.shape(x)[2], KB.shape(x)[-1]])

        f = KB.conv2d(x, kernel=self.kernel_f, strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = KB.bias_add(f, self.bias_f)

        g = KB.conv2d(x, kernel=self.kernel_g, strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = KB.bias_add(g, self.bias_g)

        h = KB.conv2d(x, kernel=self.kernel_h, strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = KB.bias_add(h, self.bias_h)

        # s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        g_ = hw_flatten(g)  # [bs,N,c']
        f_ = hw_flatten(f)  # [bs,N,c']
        f_t = KB.permute_dimensions(f_, pattern=(0, 2, 1))  # [bs,c',N]
        s = KB.batch_dot(g_, f_t)  # [bs, N, N]
        beta = KB.softmax(s)  # attention map

        o = KB.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = KB.reshape(o, shape=KB.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape
