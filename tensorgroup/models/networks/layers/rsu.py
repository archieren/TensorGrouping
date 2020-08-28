import tensorflow as tf  # TF 2.0

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

class RBC(KL.Layer):
    """
    The relu(bn(conv)) layer

    """
    def __init__(self,filters=3, kernel_size=3, padding='same', dilation_rate=1, **kwargs):
        super(RBC, self).__init__(**kwargs)
        self.filters = filters
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size


    def build(self, input_shape):
        super(RBC, self).build(input_shape)
        self.built = True

    def call(self, x):
        conv = KL.Conv2D(self.filters
                         , kernel_size=self.kernel_size
                         , padding=self.padding
                         , dilation_rate= self.dilation_rate
                         )(x)
        bn   = KL.BatchNormalization()(conv)
        relu = KL.Activation('relu')(bn)
        return relu

def down_RBC_with(rbc, down_in,pooling=True):
    hor_out=  rbc(down_in)
    if pooling :
        down_out  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_out)
    else:
        down_out  =  hor_out
    return hor_out, down_out

def up_RBC_with(rbc, hor_in, up_in, upsampling=True):
    up_out = KL.concatenate([hor_in, up_in])
    up_out = rbc(up_out)
    if upsampling:
        up_out = KL.UpSampling2D(size=2, interpolation='bilinear')(up_out)
    return up_out

class RSU7(KL.Layer):
    """
    The Residual U-Block 4
    """
    def __init__(self, filters=3, mid_filters=12, **kwargs):
        super(RSU7,self).__init__(**kwargs)
        self.filters=filters
        self.mid_filters=mid_filters

    def build(self, input_shape):
        super(RSU7,self).build(input_shape)
        self.built = True

    def call(self, x):
        down_x = x
        hor_x_0, down_x = down_RBC_with(RBC(filters=self.filters,         dilation_rate=1), down_x, pooling=False)

        hor_x_1, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x)
        hor_x_2, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x)
        hor_x_3, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x)
        hor_x_4, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x)
        hor_x_5, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x)
        hor_x_6, down_x = down_RBC_with(RBC(filters=self.mid_filters,     dilation_rate=1), down_x, pooling=False)

        up_x    =  RBC(filters=self.mid_filters, dilation_rate=2)(down_x)

        #up_x    =  KL.concatenate([hor_x_6, up_x]) ;up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x)
        up_x  = up_RBC_with(RBC(filters=self.mid_filters, dilation_rate=1), hor_x_6, up_x)
        up_x  = up_RBC_with(RBC(filters=self.mid_filters, dilation_rate=1), hor_x_5, up_x)
        up_x  = up_RBC_with(RBC(filters=self.mid_filters, dilation_rate=1), hor_x_4, up_x)
        up_x  = up_RBC_with(RBC(filters=self.mid_filters, dilation_rate=1), hor_x_3, up_x)
        up_x  = up_RBC_with(RBC(filters=self.mid_filters, dilation_rate=1), hor_x_2, up_x)
        up_x  = up_RBC_with(RBC(filters=self.filters,     dilation_rate=1), hor_x_1, up_x, upsampling=False)

        return  hor_x_0+up_x

class RSU6(KL.Layer):
    """
    The Residual U-Block 4
    """
    def __init__(self, filters=3, mid_filters=12, **kwargs):
        super(RSU6,self).__init__(**kwargs)
        self.filters=filters
        self.mid_filters=mid_filters

    def build(self, input_shape):
        super(RSU6,self).build(input_shape)
        self.built = True

    def call(self, x):
        hor_x_0 =  RBC(filters=self.filters,     dilation_rate=1)(x)     ;down_x  = hor_x_0
        hor_x_1 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_1)
        hor_x_2 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_2)
        hor_x_3 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_3)
        hor_x_4 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_4)
        hor_x_5 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  hor_x_5  # 出于形式的需要
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=2)(down_x);up_x =  KL.concatenate([hor_x_5, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_4, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_3, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_2, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_1, up_x])
        up_x    =  RBC(filters=self.filters,     dilation_rate=1)(up_x)

        return  hor_x_0+up_x

class RSU5(KL.Layer):
    """
    The Residual U-Block 4
    """
    def __init__(self, filters=3, mid_filters=12, **kwargs):
        super(RSU5,self).__init__(**kwargs)
        self.filters=filters
        self.mid_filters=mid_filters

    def build(self, input_shape):
        super(RSU5,self).build(input_shape)
        self.built = True

    def call(self, x):
        hor_x_0 =  RBC(filters=self.filters,     dilation_rate=1)(x)     ;down_x  = hor_x_0
        hor_x_1 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_1)
        hor_x_2 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_2)
        hor_x_3 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_3)
        hor_x_4 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  hor_x_4  # 出于形式的需要
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=2)(down_x);up_x =  KL.concatenate([hor_x_4, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_3, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_2, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_1, up_x])
        up_x    =  RBC(filters=self.filters,     dilation_rate=1)(up_x)

        return  hor_x_0+up_x

class RSU4(KL.Layer):
    """
    The Residual U-Block 4
    """
    def __init__(self, filters=3, mid_filters=12, **kwargs):
        super(RSU4,self).__init__(**kwargs)
        self.filters=filters
        self.mid_filters=mid_filters

    def build(self, input_shape):
        super(RSU4,self).build(input_shape)
        self.built = True

    def call(self, x):
        hor_x_0 =  RBC(filters=self.filters,     dilation_rate=1)(x)     ;down_x  = hor_x_0
        hor_x_1 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_1)
        hor_x_2 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  KL.MaxPool2D(pool_size=2, strides=2)(hor_x_2)
        hor_x_3 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x  =  hor_x_3  # 出于形式的需要
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=2)(down_x);up_x =  KL.concatenate([hor_x_3, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_2, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=1)(up_x)  ;up_x =  KL.UpSampling2D(size=2, interpolation='bilinear')(up_x);up_x=KL.concatenate([hor_x_1, up_x])
        up_x    =  RBC(filters=self.filters,     dilation_rate=1)(up_x)

        return  hor_x_0+up_x

class RSU4F(KL.Layer):
    """
    The Residual U-Block
    """
    def __init__(self, filters=3, mid_filters=12, **kwargs):
        super(RSU4F,self).__init__(**kwargs)
        self.filters=filters
        self.mid_filters=mid_filters

    def build(self, input_shape):
        super(RSU4F,self).build(input_shape)
        self.built = True

    def call(self, x):
        hor_x_0 =  RBC(filters=self.filters,     dilation_rate=1)(x)     ;down_x=hor_x_0
        hor_x_1 =  RBC(filters=self.mid_filters, dilation_rate=1)(down_x);down_x=hor_x_1
        hor_x_2 =  RBC(filters=self.mid_filters, dilation_rate=2)(down_x);down_x=hor_x_2
        hor_x_3 =  RBC(filters=self.mid_filters, dilation_rate=4)(down_x);down_x=hor_x_3
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=8)(down_x);up_x=KL.concatenate([hor_x_3, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=4)(up_x)  ;up_x=KL.concatenate([hor_x_2, up_x])
        up_x    =  RBC(filters=self.mid_filters, dilation_rate=2)(up_x)  ;up_x=KL.concatenate([hor_x_0, up_x])
        up_x    =  RBC(filters=self.filters,     dilation_rate=1)(up_x)

        return  hor_x_0+up_x
