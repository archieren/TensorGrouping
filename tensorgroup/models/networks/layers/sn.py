import tensorflow as tf  # TF 2.0

KB = tf.keras.backend

def power_iteration(W, u, rounds=1):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    _u = u
    for i in range(rounds):
        _v = KB.l2_normalize(KB.dot(_u, W))
        _u = KB.l2_normalize(KB.dot(_v, KB.transpose(W)))

    sigma = KB.sum(KB.dot(KB.dot(_u, W), KB.transpose(_v)))
    return sigma, _u, _v


class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        self.iteration = 1
        self.eps = 1e-12
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        # self.w = self.layer.kernel
        self.w = self.add_weight(name='sn_w',
                                 shape=self.layer.kernel.shape,  # 采取这样的方式，可能更合乎原论文的意思。
                                 trainable=False,
                                 dtype=self.layer.kernel.dtype)
        # With shape [KH, KW, Cin, Cout] or [H, W]
        # kernel.shape 一般是 [filter_height, filter_width, in_channels, out_channels]


        self.u = self.add_weight(shape=(1, self.w.shape[-1]), # [1, Cout] or [1, W]
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=True):
        if training: self.update_weights()
        output = self.layer(inputs)
        if training: self.restore_weights()
        return output

    def update_weights(self):
        self.w.assign(self.layer.kernel) 
        w_reshaped = tf.reshape(self.w, [self.w.shape[-1], -1]) # [Cout, KH*KW*Cin] or [W, H]
        sigma, u_hat, _ = power_iteration(w_reshaped, self.u)
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)
