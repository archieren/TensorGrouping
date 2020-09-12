import tensorflow as tf  # TF 2.0
from tensorflow.python.keras.utils import tf_utils

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


class SN_Attention(KL.Layer):
    def __init__(self, ch, spectral_normalization=True, **kwargs):
        super(SN_Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels
        self.spectral_normalization = spectral_normalization

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h, initializer='glorot_uniform', name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,), initializer='zeros', name='bias_h')
        ###
        self.u_f = self.add_weight(shape=tuple([1, self.filters_f_g]), initializer='random_uniform', name="sn_estimate_u_f", trainable=False)  # [1, out_channels]
        self.u_g = self.add_weight(shape=tuple([1, self.filters_f_g]), initializer='random_uniform', name="sn_estimate_u_g", trainable=False)  # [1, out_channels]
        self.u_h = self.add_weight(shape=tuple([1, self.filters_h]), initializer='random_uniform', name="sn_estimate_u_h", trainable=False)  # [1, out_channels]

        super(SN_Attention, self).build(input_shape)  # 这是必须的。
        # Set input spec.
        # self.input_spec = InputSpec(ndim=4,axes={3: input_shape[-1]})
        self.built = True  # 这是必须的

    def compute_spectral_normal(self, s_kernel, s_u, training):
        # Spectrally Normalized Weight
        def power_iteration(W, u, rounds=1):
            '''
            Accroding the paper, we only need to do power iteration one time.
            '''
            _u = u
            for i in range(rounds):
                _v = KB.l2_normalize(KB.dot(_u, W))
                _u = KB.l2_normalize(KB.dot(_v, KB.transpose(W)))

            W_sn = KB.sum(KB.dot(KB.dot(_u, W), KB.transpose(_v)))
            return W_sn, _u, _v

        if self.spectral_normalization:
            W_shape = s_kernel.shape.as_list()
            out_dim = W_shape[-1]
            W_mat = KB.reshape(s_kernel, [out_dim, -1])  # [out_c, N]
            sigma, u, _ = power_iteration(W_mat, s_u)

            def true_fn():
                s_u.assign(u)
                pass

            def false_fn():
                pass

            training_value = tf_utils.constant_value(training)
            if training_value is not None:
                tf_utils.smart_cond(training, true_fn, false_fn)
            return s_kernel / sigma
        else:
            return s_kernel

    def call(self, x, training=None):
        if training is None:
            training = KB.learning_phase()

        def hw_flatten(x):
            return KB.reshape(x, shape=[KB.shape(x)[0], KB.shape(x)[1]*KB.shape(x)[2], KB.shape(x)[-1]])

        f = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_f, s_u=self.u_f, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c']
        f = KB.bias_add(f, self.bias_f)

        g = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_g, s_u=self.u_g, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c']
        g = KB.bias_add(g, self.bias_g)

        h = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_h, s_u=self.u_h, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c]
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
