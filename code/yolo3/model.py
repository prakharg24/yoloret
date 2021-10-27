"""YOLO_v3 Model Defined in Keras."""

from yolo3.enums import BOX_LOSS
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
# import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typing import List, Tuple
from yolo3.utils import compose,do_giou_calculate
from yolo3.override import mobilenet_v2, mobilenet
from yolo3.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D, darknet_body
from yolo3.efficientnet import EfficientNetB3, EfficientNetB0, MBConvBlock, get_model_params, BlockArgs, MBConvBlockSpatial
from yolo3.train import AdvLossModel

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

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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

def make_last_layers_efficientnet_lite(x, block_args, global_params, quantize=False):
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
                    drop_connect_rate=global_params.drop_connect_rate, quantize=quantize))(x)
    y = compose(
        tf.keras.layers.Conv2D(block_args.output_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y

class WeightedSum(tf.keras.layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(
            name='alpha',
            shape=(4,),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return self.a[0] * model_outputs[0] + self.a[1] * model_outputs[1] + self.a[2] * model_outputs[2] + self.a[3] * model_outputs[3]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def downsample_layer(x, stride=2):
    # return tf.keras.layers.DepthwiseConv2D(5,
    #                                 padding='same',
    #                                 use_bias=False,
    #                                 strides=stride)(x)
    return tf.keras.layers.MaxPooling2D((stride, stride))(x)

def rfcr_module(inp_arr):
    b1c = inp_arr[0]
    b2c = inp_arr[1]
    b3c = inp_arr[2]
    b4c = inp_arr[3]

    b1c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False)(b1c)
    b2c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False)(b2c)
    b3c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False)(b3c)
    b4c = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', use_bias=False)(b4c)

    bc = WeightedSum()([tf.keras.layers.UpSampling2D()(b1c), b2c, downsample_layer(b3c), b4c])

    bc = MobilenetSeparableConv2D(96,
                             kernel_size=(5, 5),
                             use_bias=False,
                             padding='same')(bc)

    b1 = tf.keras.layers.Concatenate()([inp_arr[0], downsample_layer(bc)])
    b2 = tf.keras.layers.Concatenate()([inp_arr[1], bc])
    b3 = tf.keras.layers.Concatenate()([inp_arr[2], tf.keras.layers.UpSampling2D()(bc)])

    return b1, b2, b3

def yolov3_body(inputs, model_name, num_anchors, **kwargs):
    _, global_params, input_shape = get_model_params('efficientnet-b0', kwargs)
    num_classes = global_params.num_classes
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    quantize = False
    if(model_name=='mobilenetv2x75'):
        backbone = mobilenet_v2(default_batchnorm_momentum=0.9, alpha=0.75, input_tensor=inputs, include_top=False, weights=None)
        backbone_transfer = mobilenet_v2(default_batchnorm_momentum=0.9, alpha=0.75, input_tensor=inputs, include_top=False, weights='imagenet')
        layer_list = ['out_relu', 'block_16_project_BN', 'block_15_add', 'block_14_add', 'block_13_project_BN', 'block_12_add', 'block_11_add',
                      'block_10_project_BN', 'block_9_add', 'block_8_add', 'block_7_add', 'block_6_project_BN', 'block_5_add', 'block_4_add',
                      'block_3_project_BN', 'block_2_add', 'block_1_project_BN']
        # b1 = backbone.output
        b1 = backbone.get_layer('block_15_add').output
        b2 = backbone.get_layer('block_12_add').output
        b3 = backbone.get_layer('block_5_add').output
        b4 = backbone.get_layer('block_2_add').output
        b4 = downsample_layer(b4, stride=4)

    elif(model_name=='mobilenetv2x14'):
        backbone = mobilenet_v2(default_batchnorm_momentum=0.9, alpha=1.4, input_tensor=inputs, include_top=False, weights=None)
        backbone_transfer = mobilenet_v2(default_batchnorm_momentum=0.9, alpha=1.4, input_tensor=inputs, include_top=False, weights='imagenet')
        layer_list = ['out_relu', 'block_16_project_BN', 'block_15_add', 'block_14_add', 'block_13_project_BN', 'block_12_add', 'block_11_add',
                      'block_10_project_BN', 'block_9_add', 'block_8_add', 'block_7_add', 'block_6_project_BN', 'block_5_add', 'block_4_add',
                      'block_3_project_BN', 'block_2_add', 'block_1_project_BN']
        # b1 = backbone.output
        b1 = backbone.get_layer('block_15_add').output
        b2 = backbone.get_layer('block_12_add').output
        b3 = backbone.get_layer('block_5_add').output
        b4 = backbone.get_layer('block_2_add').output
        b4 = downsample_layer(b4, stride=4)

    elif(model_name=='efficientnetb3'):
        backbone = EfficientNetB3(include_top=False, weights=None, input_shape=(input_shape, input_shape, 3), input_tensor=inputs)
        backbone_transfer = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(input_shape, input_shape, 3), input_tensor=inputs)
        layer_list = ['swish_77', 'add_18', 'batch_normalization_73', 'add_17', 'add_16', 'add_15', 'add_14', 'add_13',
                      'batch_normalization_55', 'add_12', 'add_11', 'add_10', 'add_9', 'batch_normalization_40',
                      'add_8', 'add_7', 'add_6', 'add_5', 'batch_normalization_25', 'add_4', 'add_3', 'batch_normalization_16',
                      'add_2', 'add_1', 'batch_normalization_7', 'add', 'batch_normalization_2']
        # b1 = backbone.output
        b1 = backbone.get_layer('add_17').output
        b2 = backbone.get_layer('add_12').output
        b3 = backbone.get_layer('add_4').output
        b4 = backbone.get_layer('add_2').output
        b4 = downsample_layer(b4, stride=4)

    end_layer = layer_list[0]
    for i, l1 in enumerate(backbone.layers):
        backbone.layers[i].set_weights(backbone_transfer.layers[i].get_weights())
        backbone.layers[i].trainable = False
        print(backbone.layers[i].name)
        if(backbone.layers[i].name==end_layer):
            break

    b1, b2, b3 = rfcr_module([b1, b2, b3, b4])

    panet = True
    fpn = True
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=num_anchors * (num_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    x, y1 = make_last_layers_efficientnet_lite(b1, block_args,
                                          global_params, quantize=quantize)
    if(panet):
        y1 = x
    x = compose(
        tf.keras.layers.Conv2D(256,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_20_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_20_BN'),
        tf.keras.layers.ReLU(6., name='block_20_relu6'))(x)

    if(fpn):
        if(quantize):
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.UpSampling2D(), quantize_config=quantize_noop)(x)
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Concatenate(), quantize_config=quantize_noop)([x, b2])
        else:
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Concatenate()([x, b2])
    else:
        x = b2
    block_args = block_args._replace(input_filters=256)
    x, y2 = make_last_layers_efficientnet_lite(x, block_args, global_params, quantize=quantize)
    if(panet):
        y2 = x
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'))(x)

    if(fpn):
        if(quantize):
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.UpSampling2D(), quantize_config=quantize_noop)(x)
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Concatenate(), quantize_config=quantize_noop)([x, b3])
        else:
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Concatenate()([x, b3])
    else:
        x = b3
    block_args = block_args._replace(input_filters=128)
    x, y3 = make_last_layers_efficientnet_lite(x, block_args, global_params, quantize=quantize)
    if(panet):
        y3 = x

    if(panet):
        c1 = y1
        c2 = y2
        c3 = y3

        block_args = BlockArgs(kernel_size=3,
                               num_repeat=1,
                               input_filters=128,
                               output_filters=num_anchors * (num_classes + 5),
                               expand_ratio=1,
                               id_skip=True,
                               se_ratio=0.25,
                               strides=[1, 1])
        x, y3 = make_last_layers_efficientnet_lite(c3, block_args,
                                              global_params, quantize=quantize)
        x = compose(
            tf.keras.layers.Conv2D(128,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=channel_axis,
                                               momentum=0.9),
            tf.keras.layers.ReLU(6.))(x)

        if(quantize):
            x = downsample_layer(x)
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Concatenate(), quantize_config=quantize_noop)([x, c2])
        else:
            x = downsample_layer(x)
            x = tf.keras.layers.Concatenate()([x, c2])
        block_args = block_args._replace(input_filters=256)
        x, y2 = make_last_layers_efficientnet_lite(x, block_args, global_params, quantize=quantize)
        x = compose(
            tf.keras.layers.Conv2D(256,
                                   kernel_size=1,
                                   padding='same',
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(axis=channel_axis,
                                               momentum=0.9),
            tf.keras.layers.ReLU(6.))(x)

        if(quantize):
            x = downsample_layer(x)
            x = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Concatenate(), quantize_config=quantize_noop)([x, c1])
        else:
            x = downsample_layer(x)
            x = tf.keras.layers.Concatenate()([x, c1])
        block_args = block_args._replace(input_filters=512)
        x, y1 = make_last_layers_efficientnet_lite(x, block_args, global_params, quantize=quantize)


    if(quantize):
        y1 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
            tf.shape(y)[0], tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
                                    name='y1'), quantize_config=quantize_noop)(y1)

        y2 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
            tf.shape(y)[0], tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
                                    name='y2'), quantize_config=quantize_noop)(y2)
        y3 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
            tf.shape(y)[0], tf.shape(y)[1],
            tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
                                    name='y3'), quantize_config=quantize_noop)(y3)
    else:
        y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        tf.shape(y)[0], tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y1')(y1)

        y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        tf.shape(y)[0], tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y2')(y2)
        y3 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        tf.shape(y)[0], tf.shape(y)[1],
        tf.shape(y)[2], num_anchors, num_classes + 5
        ]),
        name='y3')(y3)

    return AdvLossModel(backbone.inputs, [y1, y2, y3])

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

@tf.function
def sigmoid_focal_crossentropy(
    y_true,
    y_pred,
    alpha = 0.25,
    gamma = 2.0,
    from_logits = False,
) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce
    # return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

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
        # class_loss = object_mask * sigmoid_focal_crossentropy(
        #     y_true=true_class_probs, y_pred=yolo_output[..., 5:], from_logits=True)
        class_loss = tf.reduce_sum(class_loss) / mf
        confidence_loss = tf.reduce_sum(confidence_loss) / mf

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
