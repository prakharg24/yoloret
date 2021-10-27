"""YOLO_v3 Model Defined in Keras."""

from yolo3.enums import BOX_LOSS
import numpy as np
import tensorflow.compat.v1 as tf
import keras.backend as K
from typing import List, Tuple
from yolo3.utils import compose,do_giou_calculate
from yolo3.override import mobilenet_v2, mobilenet
from yolo3.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D, darknet_body
from yolo3.efficientnet import EfficientNetB4, EfficientNetB0, MBConvBlock, get_model_params, BlockArgs
from yolo3.train import AdvLossModel


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def darknet_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    if not hasattr(inputs, '_keras_history'):
        inputs = tf.keras.layers.Input(tensor=inputs)
    darknet = darknet_body(inputs, include_top=False)
    x, y1 = make_last_layers(darknet.output, 512,
                             num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y1')(y1)
    y2 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y2')(y2)
    y3 = tf.keras.layers.Lambda(
        lambda y: tf.reshape(y, [
            -1,
            tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y3')(y3)
    return AdvLossModel(inputs, [y1, y2, y3])


def MobilenetSeparableConv2D(filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    return compose(
        tf.keras.layers.DepthwiseConv2D(kernel_size,
                                        padding=padding,
                                        use_bias=use_bias,
                                        strides=strides),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(filters,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.))

def MobilenetSeparableConv2D_lite(filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    reduction_ratio = 8
    return compose(
        tf.keras.layers.DepthwiseConv2D(kernel_size,
                                        padding=padding,
                                        use_bias=use_bias,
                                        strides=strides),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(filters//reduction_ratio,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1),
        tf.keras.layers.Conv2D(filters,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.))

def make_x_layers_mobilenet(x, id, num_filters):
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 1) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 1) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 1) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 2) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 2) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 2) + '_relu6'))(x)

    return x

def make_x_layers_mobilenet_lite(x, id, num_filters):
    reduction_ratio = 8
    x = compose(
        MobilenetSeparableConv2D_lite(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters//reduction_ratio,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id) + '_conv'),
        tf.keras.layers.Conv2D(num_filters,
                              kernel_size=1,
                              padding='same',
                              use_bias=False,
                              name='block_' + str(id) + '_conv_pep'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'))(x)

    return x

def make_y_layers_mobilenet(x, id, num_filters, out_filters):
    y = compose(
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(out_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)

    return y

def make_y_layers_mobilenet_lite(x, id, num_filters, out_filters):
    y = compose(
        MobilenetSeparableConv2D_lite(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(out_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)

    return y

def make_last_layers_mobilenet(x, id, num_filters, out_filters):
    x = make_x_layers_mobilenet(x, id, num_filters)
    y = make_y_layers_mobilenet(x, id, num_filters, out_filters)

    return x, y

def make_last_layers_mobilenet_lite(x, id, num_filters, out_filters):
    x = make_x_layers_mobilenet_lite(x, id, num_filters)
    y = make_y_layers_mobilenet_lite(x, id, num_filters, out_filters)

    return x, y

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)
    return compose(
        tf.keras.layers.Conv2D(last_block_filters,
                               kernel,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.))


def mobilenetv2_yolo_body(inputs, num_anchors, num_classes, alpha=1.0, model_wrapper=AdvLossModel):
    # import tensorflow as tf
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')
    # mobilenetv2 = tf.keras.models.load_model('/home/prakhar/mobilenetv2-yolov3/xalogic/mobilenet_v1_base_7.h5')
    # print(mobilenetv2.summary())
    # exit()
    print(len(mobilenetv2.layers))
    lay_12 = 'block_12_project_BN'
    # lay_12 = 'conv_pw_11_relu'
    lay_5 = 'block_5_project_BN'
    # lay_5 = 'conv_pw_5_relu'

    b1 = mobilenetv2.output
    b2 = mobilenetv2.get_layer(lay_12).output
    b3 = mobilenetv2.get_layer(lay_5).output

    bc = tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(b1), b2, tf.keras.layers.MaxPooling2D()(b3)])

    bc = MobilenetSeparableConv2D_lite(384,
                             kernel_size=(3, 3),
                             use_bias=False,
                             padding='same')(bc)

    b1 = tf.keras.layers.Concatenate()([b1, tf.keras.layers.MaxPooling2D()(bc)])
    b2 = tf.keras.layers.Concatenate()([b2, bc])
    b3 = tf.keras.layers.Concatenate()([b3, tf.keras.layers.UpSampling2D()(bc)])

    x1, y1 = make_last_layers_mobilenet_lite(b1, 17, 512,
                                       num_anchors * (num_classes + 5))
    x1 = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x1)
    x2 = tf.keras.layers.Concatenate()([
        x1,
        MobilenetConv2D(
            (1, 1), alpha,
            384)(b2)
    ])
    # x2 = MobilenetConv2D(
    #         (1, 1), alpha,
    #         384)(mobilenetv2.get_layer('block_12_project_BN').output)
    x2, y2 = make_last_layers_mobilenet_lite(x2, 21, 256,
                                       num_anchors * (num_classes + 5))
    x2 = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x2)
    x3 = tf.keras.layers.Concatenate()([
        x2,
        MobilenetConv2D((1, 1), alpha,
                        128)(b3)
    ])
    # x3 = MobilenetConv2D(
    #         (1, 1), alpha,
    #         128)(mobilenetv2.get_layer('block_5_project_BN').output)
    x3, y3 = make_last_layers_mobilenet_lite(x3, 25, 128,
                                       num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)

    return model_wrapper(mobilenetv2.inputs, [y1, y2, y3])
    # return model_wrapper(mobilenetv2.inputs, [y1, y2])

def skynet_reorg_layer(x):
    stride = 2

    B, H, W, C = x.shape
    ws = stride
    hs = stride

    x = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [-1, H//hs, hs, W//ws, ws, C]))(x)
    x = tf.keras.layers.Lambda(lambda y: tf.transpose(y, perm=[0, 1, 3, 2, 4, 5]))(x)
    x = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [-1, H//hs, W//ws, hs*ws, C]))(x)
    x = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [-1, H//hs, W//ws, hs*ws*C]))(x)

    return x
    # return tf.keras.layers.MaxPooling2D()(x)

def skynet_body(inputs, num_anchors, num_classes, alpha=1.0, model_wrapper=AdvLossModel):
    import tensorflow as tf

    x = MobilenetSeparableConv2D(48, kernel_size=(3, 3), use_bias=False, padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = MobilenetSeparableConv2D(96, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = MobilenetSeparableConv2D(192, kernel_size=(3, 3), use_bias=False, padding='same')(x)

    x_short = skynet_reorg_layer(x)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = MobilenetSeparableConv2D(384, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = MobilenetSeparableConv2D(512, kernel_size=(3, 3), use_bias=False, padding='same')(x)

    x = tf.keras.layers.Concatenate()([x_short, x])

    x = MobilenetSeparableConv2D(96, kernel_size=(3, 3), use_bias=False, padding='same')(x)

    y1 = tf.keras.layers.Conv2D(num_anchors * (num_classes + 5), kernel_size=1, padding='same', use_bias=False)(x)

    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)

    return model_wrapper(inputs, [y1])


def mobilenetv2_skynet_body(inputs, num_anchors, num_classes, alpha=1.0, model_wrapper=AdvLossModel):
    import tensorflow as tf
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')
    print(len(mobilenetv2.layers))
    # print(mobilenetv2.summary())
    # exit()

    # x4 = skynet_reorg_layer(mobilenetv2.get_layer('block_2_project_BN').output)
    # # x4 = compose(
    # #     tf.keras.layers.Conv2D(128,
    # #                            kernel_size=1,
    # #                            padding='same',
    # #                            use_bias=False,
    # #                            name='block_20_conv'),
    # #     tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
    # #     tf.keras.layers.ReLU(6., name='block_20_relu6'))(x4)
    # # x3 = tf.keras.layers.Concatenate()([
    # #     x4,
    # #     MobilenetConv2D(
    # #         (1, 1), alpha,
    # #         128)(mobilenetv2.get_layer('block_5_project_BN').output)
    # # ])
    # x3 = tf.keras.layers.Concatenate()([
    #     x4,
    #     mobilenetv2.get_layer('block_5_project_BN').output
    # ])
    x3 = mobilenetv2.get_layer('block_5_project_BN').output

    x3, y3 = make_last_layers_mobilenet(x3, 25, 128,
                                       num_anchors * (num_classes + 5))

    x3 = skynet_reorg_layer(x3)
    # x3 = compose(
    #     tf.keras.layers.Conv2D(256,
    #                            kernel_size=1,
    #                            padding='same',
    #                            use_bias=False,
    #                            name='block_24_conv'),
    #     tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
    #     tf.keras.layers.ReLU(6., name='block_24_relu6'))(x3)
    # x2 = tf.keras.layers.Concatenate()([
    #     x3,
    #     MobilenetConv2D(
    #         (1, 1), alpha,
    #         384)(mobilenetv2.get_layer('block_12_project_BN').output)
    # ])
    x2 = tf.keras.layers.Concatenate()([
        x3,
        mobilenetv2.get_layer('block_12_project_BN').output
    ])

    x2, y2 = make_last_layers_mobilenet(x2, 21, 256,
                                       num_anchors * (num_classes + 5))

    x2 = skynet_reorg_layer(x2)
    # x2 = compose(
    #     tf.keras.layers.Conv2D(512,
    #                            kernel_size=1,
    #                            padding='same',
    #                            use_bias=False),
    #     tf.keras.layers.BatchNormalization(momentum=0.9),
    #     tf.keras.layers.ReLU(6.))(x2)
    # x1 = tf.keras.layers.Concatenate()([
    #     x2,
    #     MobilenetConv2D(
    #         (1, 1), alpha,
    #         1344)(mobilenetv2.output)
    # ])
    x1 = tf.keras.layers.Concatenate()([
        x2,
        mobilenetv2.output
    ])

    x1, y1 = make_last_layers_mobilenet(x1, 17, 512,
                                       num_anchors * (num_classes + 5))

    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors, num_classes + 5])(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors*(num_classes + 5)])(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors, num_classes + 5])(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors*(num_classes + 5)])(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)
    # y3 = tf.keras.layers.Reshape([-1, 28, 28, num_anchors, num_classes + 5])(y3)
    return model_wrapper(mobilenetv2.inputs, [y1, y2, y3])
    # return model_wrapper(mobilenetv2.inputs, [y1, y2])

def mobilenetv2_scarf_body(inputs, num_anchors, num_classes, alpha=1.0, model_wrapper=AdvLossModel):
    import tensorflow as tf
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')
    print(len(mobilenetv2.layers))

    x1 = mobilenetv2.output
    x2 = mobilenetv2.get_layer('block_12_project_BN').output
    x3 = mobilenetv2.get_layer('block_5_project_BN').output

    x1r = tf.keras.layers.UpSampling2D()(x1)
    x2r = x2
    x3r = tf.keras.layers.MaxPooling2D()(x3)
    xr = tf.keras.layers.Concatenate()([x1r, x2r, x3r])

    xr = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'))(xr)

    x1 = tf.keras.layers.Concatenate()([
        x1,
        tf.keras.layers.MaxPooling2D()(xr)
    ])

    x2 = tf.keras.layers.Concatenate()([
        x2,
        xr
    ])

    x3 = tf.keras.layers.Concatenate()([
        x3,
        tf.keras.layers.UpSampling2D()(xr)
    ])

    x1, y1 = make_last_layers_mobilenet(x1, 17, 512,
                                       num_anchors * (num_classes + 5))
    x2, y2 = make_last_layers_mobilenet(x2, 21, 256,
                                       num_anchors * (num_classes + 5))
    x3, y3 = make_last_layers_mobilenet(x3, 25, 128,
                                       num_anchors * (num_classes + 5))

    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors, num_classes + 5])(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors*(num_classes + 5)])(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors, num_classes + 5])(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors*(num_classes + 5)])(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)
    # y3 = tf.keras.layers.Reshape([-1, 28, 28, num_anchors, num_classes + 5])(y3)
    return model_wrapper(mobilenetv2.inputs, [y1, y2, y3])
    # return model_wrapper(mobilenetv2.inputs, [y1, y2])

def conv1x1(input, output_channels, stride=1, bn=True):
    # 1x1 convolution without padding
    if bn == True:
        return compose(
            tf.keras.layers.Conv2D(output_channels,
                                   kernel_size=1,
                                   strides=stride,
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.ReLU(6.))(input)
    else:
        return tf.keras.layers.Conv2D(output_channels,
                               kernel_size=1,
                               strides=stride,
                               use_bias=False)(input)

def conv3x3(input, output_channels, stride=1, bn=True):
    # 3x3 convolution with padding=1
    if bn == True:
        return compose(
            tf.keras.layers.Conv2D(output_channels,
                                   kernel_size=3,
                                   strides=stride,
                                   padding='same',
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.ReLU(6.))(input)
    else:
        return tf.keras.layers.Conv2D(output_channels,
                               kernel_size=3,
                               strides=stride,
                               padding='same',
                               use_bias=False)(input)

def sepconv3x3(input, input_channels, output_channels, stride=1, expand_ratio=1):
    return compose(
        tf.keras.layers.Conv2D(input_channels * expand_ratio,
                               kernel_size=1,
                               strides=1,
                               use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(6.),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                        padding='same',
                                        use_bias=False,
                                        strides=stride),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(output_channels,
                               kernel_size=1,
                               strides=1,
                               use_bias=False),
        tf.keras.layers.BatchNormalization(momentum=0.9))(input)

def EP(input, input_channels, output_channels, stride=1):
    # print("EP : ",input.shape)
    use_res_connect = stride == 1 and input_channels == output_channels

    out = sepconv3x3(input, input_channels, output_channels, stride=stride)
    if(use_res_connect):
        return input + out
    else:
        return out

def PEP(input, input_channels, output_channels, mid_channels, stride=1):
    # print("PEP : ",input.shape)
    use_res_connect = stride == 1 and input_channels == output_channels

    out = conv1x1(input, mid_channels)
    out = sepconv3x3(out, input_channels, output_channels, stride=stride)
    if(use_res_connect):
        return input + out
    else:
        return out

def FCA(input, output_channels, reduction_ratio):
    hidden_channels = output_channels // reduction_ratio

    b, _, _, c = input.shape
    out = tf.keras.layers.GlobalAveragePooling2D()(input)
    out = compose(
        tf.keras.layers.Dense(hidden_channels, use_bias=False),
        tf.keras.layers.ReLU(6.),
        tf.keras.layers.Dense(output_channels, use_bias=False),
        tf.keras.layers.Activation(activation='sigmoid'))(out)

    out = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [-1, 1, 1, c]))(out)

    return input*out

def nano_backbone(x):
    x = conv3x3(x, 12, stride=1) # output: 416x416x12
    x = conv3x3(x, 24, stride=2) # output: 208x208x24
    x = PEP(x, 24, 24, 7, stride=1) # output: 208x208x24
    x = EP(x, 24, 70, stride=2) # output: 104x104x70
    x = PEP(x, 70, 70, 25, stride=1) # output: 104x104x70
    x = PEP(x, 70, 70, 24, stride=1) # output: 104x104x70
    x = EP(x, 70, 150, stride=2) # output: 52x52x150
    x = PEP(x, 150, 150, 56, stride=1) # output: 52x52x150
    x = conv1x1(x, 150, stride=1) # output: 52x52x150
    x = FCA(x, 150, 8)
    x = PEP(x, 150, 150, 73, stride=1) # output: 52x52x150
    x = PEP(x, 150, 150, 71, stride=1) # output: 52x52x150

    x1 = PEP(x, 150, 150, 75, stride=1) # output: 52x52x150
    x = EP(x1, 150, 325, stride=2) # output: 26x26x325
    x = PEP(x, 325, 325, 132, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 124, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 141, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 140, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 137, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 135, stride=1) # output: 26x26x325
    x = PEP(x, 325, 325, 133, stride=1) # output: 26x26x325

    x2 = PEP(x, 325, 325, 140, stride=1) # output: 26x26x325
    x = EP(x2, 325, 545, stride=2) # output: 13x13x545
    x = PEP(x, 545, 545, 276, stride=1) # output: 13x13x545
    x = conv1x1(x, 230, stride=1) # output: 13x13x230
    x = EP(x, 230, 489, stride=1) # output: 13x13x489
    x3 = PEP(x, 489, 469, 213, stride=1) # output: 13x13x469

    return x1, x2, x3

def yolo_nano(inputs, num_anchors, num_classes, alpha=1.0, model_wrapper=AdvLossModel):
    import tensorflow as tf

    out52, out26, out13 = nano_backbone(inputs)

    x1 = conv1x1(out13, 189, stride=1) # output: 13x13x189
    x = conv1x1(x1, 105, stride=1) # output: 13x13x105
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, out26])

    x = PEP(x, 430, 325, 113, stride=1) # output: 26x26x325
    x = PEP(x, 325, 207, 99, stride=1) # output: 26x26x325

    x2 = conv1x1(x, 98, stride=1) # output: 26x26x98
    x = conv1x1(x2, 47, stride=1) # output: 13x13x105
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, out52])

    x = PEP(x, 197, 122, 58, stride=1) # output: 52x52x122
    x = PEP(x, 122, 87, 52, stride=1) # output: 52x52x87

    x3 = PEP(x, 87, 93, 47, stride=1) # output: 52x52x93
    y3 = conv1x1(x, num_anchors * (num_classes + 5), stride=1, bn=False) # output: 52x52x yolo_channels

    x2 = EP(x2, 98, 183, stride=1) # output: 26x26x183
    y2 = conv1x1(x2, num_anchors * (num_classes + 5), stride=1, bn=False) # output: 26x26x yolo_channels

    x1 = EP(x1, 189, 462, stride=1) # output: 13x13x462
    y1 = conv1x1(x1, num_anchors * (num_classes + 5), stride=1, bn=False) # output: 13x13x yolo_channels

    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors, num_classes + 5])(y1)
    # y1 = tf.keras.layers.Reshape([-1, 7, 7, num_anchors*(num_classes + 5)])(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors, num_classes + 5])(y2)
    # y2 = tf.keras.layers.Reshape([-1, 14, 14, num_anchors*(num_classes + 5)])(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)
    # y3 = tf.keras.layers.Reshape([-1, 28, 28, num_anchors, num_classes + 5])(y3)
    return model_wrapper(inputs, [y1, y2, y3])
    # return model_wrapper(mobilenetv2.inputs, [y1, y2])


def make_last_layers_efficientnet(x, block_args, global_params):
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    num_filters = block_args.input_filters * block_args.expand_ratio
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum),
        tf.keras.layers.ReLU(6.))(x)
    y = compose(
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(block_args.output_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def efficientnet_yolo_lite(inputs, model_name, num_anchors, **kwargs):
    _, global_params, input_shape = get_model_params(model_name, kwargs)
    num_classes = global_params.num_classes
    alpha=0.75
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    efficientnet = EfficientNetB0(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_shape, input_shape, 3),
                                  input_tensor=inputs)
    #print(len(efficientnet.layers))
    #print(efficientnet.summary())
    #exit()

    b1 = efficientnet.output
    b2 = efficientnet.get_layer('swish_33').output
    b3 = efficientnet.get_layer('swish_15').output

    bc = tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(b1), b2, tf.keras.layers.MaxPooling2D()(b3)])

    bc = MobilenetSeparableConv2D_lite(384,
                             kernel_size=(3, 3),
                             use_bias=False,
                             padding='same')(bc)

    b1 = tf.keras.layers.Concatenate()([b1, tf.keras.layers.MaxPooling2D()(bc)])
    b2 = tf.keras.layers.Concatenate()([b2, bc])
    b3 = tf.keras.layers.Concatenate()([b3, tf.keras.layers.UpSampling2D()(bc)])

    x1, y1 = make_last_layers_mobilenet_lite(b1, 17, 512,
                                       num_anchors * (num_classes + 5))
    x1 = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x1)
    x2 = tf.keras.layers.Concatenate()([
        x1,
        MobilenetConv2D(
            (1, 1), alpha,
            384)(b2)
    ])
    # x2 = MobilenetConv2D(
    #         (1, 1), alpha,
    #         384)(mobilenetv2.get_layer('block_12_project_BN').output)
    x2, y2 = make_last_layers_mobilenet_lite(x2, 21, 256,
                                       num_anchors * (num_classes + 5))
    x2 = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x2)
    x3 = tf.keras.layers.Concatenate()([
        x2,
        MobilenetConv2D((1, 1), alpha,
                        128)(b3)
    ])
    # x3 = MobilenetConv2D(
    #         (1, 1), alpha,
    #         128)(mobilenetv2.get_layer('block_5_project_BN').output)
    x3, y3 = make_last_layers_mobilenet_lite(x3, 25, 128,
                                       num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y1')(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y2')(y2)
    y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
    ]),
                                name='y3')(y3)

    return AdvLossModel(efficientnet.inputs, [y1, y2, y3])

def efficientnet_yolo_body(inputs, model_name, num_anchors, **kwargs):
    _, global_params, input_shape = get_model_params(model_name, kwargs)
    num_classes = global_params.num_classes
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    efficientnet = EfficientNetB0(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_shape, input_shape, 3),
                                  input_tensor=inputs)
    #print(len(efficientnet.layers))
    #print(efficientnet.summary())
    #exit()
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=num_anchors * (num_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    x, y1 = make_last_layers_efficientnet(efficientnet.output, block_args,
                                          global_params)
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    x = tf.keras.layers.Concatenate()(
        [x, efficientnet.get_layer('swish_33').output])
    x, y2 = make_last_layers_efficientnet(x, block_args, global_params)
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, efficientnet.get_layer('swish_15').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, global_params)
    y1 = tf.keras.layers.Reshape(
        (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
    y2 = tf.keras.layers.Reshape(
        (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
    y3 = tf.keras.layers.Reshape(
        (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)

    return AdvLossModel(efficientnet.inputs, [y1, y2, y3])

def yolo_head(feats: tf.Tensor,
              anchors: np.ndarray,
              input_shape: tf.Tensor,
              calc_loss: bool = False
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = tf.shape(feats)[1:3]
    # grid_shape = tf.shape(feats)[0:2]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * tf.cast(
        anchors_tensor, feats.dtype) / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    if calc_loss == True:
        return grid, box_xy, box_wh, box_confidence
    box_class_probs = tf.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy: tf.Tensor, box_wh: tf.Tensor,
                       input_shape: tf.Tensor, image_shape) -> tf.Tensor:
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype)
    image_shape = tf.cast(image_shape, box_yx.dtype)
    max_shape = tf.maximum(image_shape[0], image_shape[1])
    ratio = image_shape / max_shape
    boxed_shape = input_shape * ratio
    offset = (input_shape - boxed_shape) / 2.
    scale = image_shape / boxed_shape
    box_yx = (box_yx * input_shape - offset) * scale
    box_hw *= input_shape * scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat(
        [
            tf.clip_by_value(box_mins[..., 0:1], 0, image_shape[0]),  # y_min
            tf.clip_by_value(box_mins[..., 1:2], 0, image_shape[1]),  # x_min
            tf.clip_by_value(box_maxes[..., 0:1], 0, image_shape[0]),  # y_max
            tf.clip_by_value(box_maxes[..., 1:2], 0, image_shape[1])  # x_max
        ],
        -1)
    return boxes


def yolo_boxes_and_scores(feats: tf.Tensor, anchors: List[Tuple[float, float]],
                          num_classes: int, input_shape: Tuple[int, int],
                          image_shape, zoom_feats=None) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(
        feats, anchors, input_shape)
    if(zoom_feats is not None):
        box_xy_z, box_wh_z, box_confidence_z, box_class_probs_z = yolo_head(
            zoom_feats, anchors, input_shape)
        box_xy_z = box_xy_z*(224/416) + (416-224)/(2*416)
        box_wh_z = box_wh_z*(224/416)

        box_xy = tf.concat([box_xy, box_xy_z], -2)
        box_wh = tf.concat([box_wh, box_wh_z], -2)
        box_confidence = tf.concat([box_confidence, box_confidence_z], -2)
        box_class_probs = tf.concat([box_class_probs, box_class_probs_z], -2)
        # print(box_xy.numpy().shape)
        # print(box_wh_z.numpy().shape)
        # print(box_confidence_z.numpy().shape)
        # print(box_class_probs_z.numpy().shape)
        # exit()

    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs: List[tf.Tensor],
              anchors: np.ndarray,
              num_scales,
              num_classes: int,
              image_shape,
              max_boxes: int = 20,
              score_threshold: float = .6,
              iou_threshold: float = .5,
              zoom_outputs=None
             ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchor_mask = anchor_mask[-1*num_scales:]
    # anchor_mask = [[6, 7, 8]]


    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    # input_shape = tf.shape(yolo_outputs[0])[1:3] * 8
    boxes = []
    box_scores = []
    for l in range(num_scales):
        if(zoom_outputs is not None):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes, input_shape,
                                                    image_shape,
                                                    zoom_feats=zoom_outputs[l])
        else:
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes, input_shape,
                                                    image_shape)

        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        nms_index = tf.image.non_max_suppression(
            boxes,
            box_scores[:, c],
            max_boxes_tensor,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold)
        class_boxes = tf.gather(boxes, nms_index)
        class_box_scores = tf.gather(box_scores[:, c], nms_index)
        classes = tf.ones_like(class_box_scores, tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0, name='scores')
    classes_ = tf.concat(classes_, axis=0, name='classes')
    boxes_ = tf.cast(boxes_, tf.int32, name='boxes')
    return boxes_, scores_, classes_


class YoloEval(tf.keras.layers.Layer):

    def __init__(self,
                 anchors,
                 num_scales,
                 num_classes,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5,
                 **kwargs):
        super(YoloEval, self).__init__(**kwargs)
        self.anchors = anchors
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def call(self, yolo_outputs,image_shape,zoom_outputs=None):
        return yolo_eval(yolo_outputs, self.anchors, self.num_scales, self.num_classes,
                         image_shape, self.max_boxes, self.score_threshold,
                         self.iou_threshold, zoom_outputs=zoom_outputs)

    def get_config(self):
        config = super(YoloEval, self).get_config()
        config['anchors'] = self.anchors
        config['num_scales'] = self.num_scales
        config['num_classes'] = self.num_classes
        config['max_boxes'] = self.max_boxes
        config['score_threshold'] = self.score_threshold
        config['iou_threshold'] = self.iou_threshold

        return config


class YoloLoss(tf.keras.losses.Loss):

    def __init__(self,
                    idx,
                    anchors,
                    num_scales,
                    ignore_thresh=.5,
                    box_loss=BOX_LOSS.GIOU,
                    print_loss=True):
        super(YoloLoss, self).__init__(reduction=tf.losses.Reduction.NONE,name='yolo_loss')
        # grid_steps = [8]
        grid_steps = [32, 16, 8]
        anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchor_masks = anchor_masks[-1*num_scales:]
        # anchor_masks = [[6, 7, 8]]
        self.idx = idx
        self.ignore_thresh = ignore_thresh
        self.box_loss = box_loss
        self.print_loss = print_loss
        self.grid_step = grid_steps[self.idx]
        self.anchor = anchors[anchor_masks[idx]]

    def call(self, y_true, yolo_output):
        '''Return yolo_loss tensor

        Parameters
        ----------
        yolo_output: the output of yolo_body
        y_true: the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss

        Returns
        -------
        loss: tensor, shape=(1,)

        '''
        loss = 0
        m = tf.shape(yolo_output)[0]  # batch size, tensor
        mf = tf.cast(m, yolo_output.dtype)
        object_mask = y_true[..., 4:5]
        true_class_probs = y_true[..., 5:]
        input_shape = tf.shape(yolo_output)[1:3] * self.grid_step
        grid, pred_xy, pred_wh, box_confidence = yolo_head(
            yolo_output, self.anchor, input_shape, calc_loss=True)
        pred_max = tf.reverse(pred_xy + pred_wh / 2., [-1])
        pred_min = tf.reverse(pred_xy - pred_wh / 2., [-1])
        pred_box = tf.concat([pred_min, pred_max], -1)

        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_max = tf.reverse(true_xy + true_wh / 2., [-1])
        true_min = tf.reverse(true_xy - true_wh / 2., [-1])
        true_box = tf.concat([true_min, true_max], -1)
        true_box = tf.clip_by_value(true_box, 0, 1)
        object_mask_bool = tf.cast(object_mask, 'bool')

        masked_true_box = tf.boolean_mask(true_box, object_mask_bool[..., 0])
        iou = do_giou_calculate(
            tf.expand_dims(pred_box, -2),
            tf.expand_dims(masked_true_box, 0),
            mode='iou')
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, masked_true_box.dtype)

        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # focal_loss = focal(object_mask, box_confidence)
        confidence_loss = (object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=yolo_output[..., 4:5]) + \
                        (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                    logits=yolo_output[...,
                                                                                            4:5]) * ignore_mask)
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_class_probs, logits=yolo_output[..., 5:])
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf

        if self.box_loss == BOX_LOSS.GIOU:
            giou = do_giou_calculate(pred_box, true_box)
            giou_loss = object_mask * (1 - tf.expand_dims(giou, -1))
            giou_loss = tf.reduce_sum(giou_loss) / mf
            loss += giou_loss + confidence_loss + class_loss
            if self.print_loss:
                tf.print(str(self.idx)+':',giou_loss, confidence_loss, class_loss,tf.reduce_sum(ignore_mask))
        elif self.box_loss == BOX_LOSS.MSE:
            grid_shape = tf.cast(tf.shape(yolo_output)[1:3], y_true.dtype)
            raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
            raw_true_wh = tf.math.log(y_true[..., 2:4] /
                                    tf.cast(anchors[anchor_mask[idx]], y_true.dtype)*
                                    tf.cast(input_shape[::-1], y_true.dtype) )
            raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh,
                                                tf.zeros_like(raw_true_wh))
            box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=raw_true_xy, logits=yolo_output[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(
                raw_true_wh - yolo_output[..., 2:4])
            xy_loss = tf.reduce_sum(xy_loss) / mf
            wh_loss = tf.reduce_sum(wh_loss) / mf
            loss += xy_loss + wh_loss + confidence_loss + class_loss
            if print_loss:
                tf.print(loss, xy_loss, wh_loss, confidence_loss, class_loss,
                        tf.reduce_sum(ignore_mask))
        return loss

if True:
    run_meta = tf.RunMetadata()
    sess = tf.Session(graph=tf.get_default_graph())

    K.set_session(sess)
    inputs = tf.placeholder('float32', shape=(1,224,224,3))
    inputs = tf.keras.layers.Input(tensor=inputs)
    base_model = efficientnet_yolo_body(inputs, 'efficientnet-b0', 3, batch_norm_momentum=0.9,
                                  batch_norm_epsilon=1e-3,
                                  num_classes=num_classes,
                                  drop_connect_rate=0.2,
                                  data_format="channels_first")
    print(len(base_model.layers))
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print(flops.total_float_ops)
    exit()
