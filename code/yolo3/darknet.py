from functools import wraps
import tensorflow as tf
from yolo3.utils import compose


@wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    # darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (
        2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   tf.keras.layers.BatchNormalization(),
                   tf.keras.layers.LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
                    DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = tf.keras.layers.Add()([x, y])
    return x


def darknet_body(inputs, include_top=True, classes=1000):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(classes,
                                  activation='softmax',
                                  use_bias=True,
                                  name='Logits')(x)
    return tf.keras.Model(inputs, x)

def yolo_fastest_block(input, filters, exp_filters, strides):
    padding = 'valid' if strides==2 else 'same'
    x = compose(
        tf.keras.layers.Conv2D(exp_filters, 1, padding='same', strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1))(input)

    x = compose(
        tf.keras.layers.DepthwiseConv2D(3, padding=padding, strides=strides),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1))(x)

    x = compose(
        tf.keras.layers.Conv2D(filters, 1, padding='same', strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1))(input)

    if(strides==1):
        x = x + input
    return x


def yolo_fastest_xl(input):
    x = tf.keras.layers.Conv2D(16, 3, padding='valid', strides=2)(input)
    x = tf.keras.layers.Conv2D(16, 1, strides=1)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(8, 1, strides=1)(x)

    x = yolo_fastest_block(x, 8, 16, 1)
    x = yolo_fastest_block(x, 16, 48, 2)

    x = yolo_fastest_block(x, 16, 64, 1)
    x = yolo_fastest_block(x, 16, 64, 1)
    x = yolo_fastest_block(x, 16, 64, 2)

    x = yolo_fastest_block(x, 16, 96, 1)
    x = yolo_fastest_block(x, 16, 96, 1)
    x = yolo_fastest_block(x, 32, 96, 2)

    x = yolo_fastest_block(x, 32, 192, 1)
    x = yolo_fastest_block(x, 32, 192, 1)
    x = yolo_fastest_block(x, 32, 192, 1)
    x = yolo_fastest_block(x, 32, 192, 1)
    route2 = x
    x = yolo_fastest_block(x, 48, 192, 2)

    x = yolo_fastest_block(x, 48, 272, 1)
    x = yolo_fastest_block(x, 48, 272, 1)
    x = yolo_fastest_block(x, 48, 272, 1)
    x = yolo_fastest_block(x, 48, 272, 1)
    route1 = x
    x = yolo_fastest_block(x, 96, 272, 2)

    x = yolo_fastest_block(x, 96, 448, 1)
    x = yolo_fastest_block(x, 96, 448, 1)
    x = yolo_fastest_block(x, 96, 448, 1)
    x = yolo_fastest_block(x, 96, 448, 1)
    x = yolo_fastest_block(x, 96, 448, 1)

    x = tf.keras.layers.Conv2D(96, 1, strides=1)(x)

    b1 = tf.keras.layers.Concatenate()([route1, tf.keras.layers.UpSampling2D()(x)])
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(75, 1, strides=1)(b1)


    x = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(128, 1, strides=1)(x)
    x = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(128, 1, strides=1)(x)
    x = tf.keras.layers.Conv2D(75, 1, strides=1)(x)

    b2 = x

    b3 = tf.keras.layers.Conv2D(75, 1, strides=1)(route2)

def yolo_fastest(input):
    x = tf.keras.layers.Conv2D(8, 3, padding='valid', strides=2)(input)
    x = tf.keras.layers.Conv2D(8, 1, strides=1)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(4, 1, strides=1)(x)

    x = yolo_fastest_block(x, 4, 8, 1)
    x = yolo_fastest_block(x, 8, 24, 2)

    x = yolo_fastest_block(x, 8, 32, 1)
    x = yolo_fastest_block(x, 8, 32, 1)
    x = yolo_fastest_block(x, 8, 32, 2)

    x = yolo_fastest_block(x, 8, 48, 1)
    x = yolo_fastest_block(x, 8, 48, 1)
    x = yolo_fastest_block(x, 16, 48, 2)

    x = yolo_fastest_block(x, 16, 96, 1)
    x = yolo_fastest_block(x, 16, 96, 1)
    x = yolo_fastest_block(x, 16, 96, 1)
    x = yolo_fastest_block(x, 16, 96, 1)
    route2 = x
    x = yolo_fastest_block(x, 24, 96, 2)

    x = yolo_fastest_block(x, 24, 136, 1)
    x = yolo_fastest_block(x, 24, 136, 1)
    x = yolo_fastest_block(x, 24, 136, 1)
    x = yolo_fastest_block(x, 24, 136, 1)
    route1 = x
    x = yolo_fastest_block(x, 48, 136, 2)

    x = yolo_fastest_block(x, 48, 224, 1)
    x = yolo_fastest_block(x, 48, 224, 1)
    x = yolo_fastest_block(x, 48, 224, 1)
    x = yolo_fastest_block(x, 48, 224, 1)
    x = yolo_fastest_block(x, 48, 224, 1)

    x = tf.keras.layers.Conv2D(96, 1, strides=1)(x)

    b1 = tf.keras.layers.Concatenate()([route1, tf.keras.layers.UpSampling2D()(x)])
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(96, 1, strides=1)(b1)
    b1 = tf.keras.layers.Conv2D(75, 1, strides=1)(b1)


    x = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(128, 1, strides=1)(x)
    x = tf.keras.layers.DepthwiseConv2D(5, padding='same', strides=1)(x)
    x = tf.keras.layers.Conv2D(128, 1, strides=1)(x)
    x = tf.keras.layers.Conv2D(75, 1, strides=1)(x)

    b2 = x

    b3 = tf.keras.layers.Conv2D(75, 1, strides=1)(route2)
