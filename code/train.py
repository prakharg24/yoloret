"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import datetime
from yolo3.model import YoloEval, YoloLoss, yolov3_body
from yolo3.efficientnet import EfficientConv2DKernelInitializer, Swish, Mean
from yolo3.data import Dataset
from yolo3.enums import BACKBONE, DATASET_MODE
from yolo3.map import MAPCallback
from yolo3.utils import get_anchors, get_classes, ModelFactory
from yolo3.train import AdvLossModel
import os
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.keras.backend.set_learning_phase(1)

def train(FLAGS):
    """Train yolov3 with different backbone
    """
    prune = FLAGS['prune']
    opt = FLAGS['opt']
    backbone = FLAGS['backbone']
    log_dir = os.path.join(
        'logs',
        str(backbone).split('.')[1].lower() + '_' + str(datetime.date.today()))

    batch_size = FLAGS['batch_size']
    num_scales = FLAGS['num_scales']
    train_dataset_glob = FLAGS['train_dataset']
    val_dataset_glob = FLAGS['val_dataset']
    test_dataset_glob = FLAGS['test_dataset']
    freeze = FLAGS['freeze']
    quantize = FLAGS['quantize']
    epochs = FLAGS['epochs'][0] if freeze else FLAGS['epochs'][1]

    class_names = get_classes(FLAGS['classes_path'])
    num_classes = len(class_names)
    anchors = get_anchors(FLAGS['anchors_path'])
    input_shape = FLAGS['input_size']  # multiple of 32, hw
    model_path = FLAGS['model']
    if model_path and model_path.endswith(
            '.h5') is not True:
        model_path = tf.train.latest_checkpoint(model_path)
    lr = FLAGS['learning_rate']
    tpu_address=FLAGS['tpu_address']
    if tpu_address is not None:
        cluster_resolver=tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        tf.config.experimental_connect_to_host(cluster_resolver.master())
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy=tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        strategy = tf.distribute.MirroredStrategy(devices=FLAGS['gpus'])
    batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset_builder = Dataset(train_dataset_glob, batch_size, anchors,
                                     num_classes, input_shape, num_scales)
    train_dataset_org, train_num = train_dataset_builder.build(epochs)
    val_dataset_builder = Dataset(val_dataset_glob,
                                  batch_size,
                                  anchors,
                                  num_classes,
                                  input_shape,
                                  num_scales,
                                  mode=DATASET_MODE.VALIDATE)
    val_dataset_org, val_num = val_dataset_builder.build(epochs)
    map_callback = MAPCallback(test_dataset_glob, input_shape, anchors,
                               class_names)
    tensorboard = tf.keras.callbacks.TensorBoard(write_graph=False,
                                             log_dir=log_dir,
                                             write_images=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    period=3)
    checkpoint_quantize = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    period=1)
    checkpoint_replace = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        log_dir, 'best_scarfnet.h5'),
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    period=3)
    cos_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[1], epochs)(
            epoch).numpy(),1)
    cos_lr_freeze = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[0], epochs)(
            epoch).numpy(),1)
    cos_lr_quantize = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, _: tf.keras.experimental.CosineDecay(lr[1], 10)(
            epoch).numpy(),1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=epochs // 2,
        verbose=1)

    loss = [YoloLoss(idx, anchors, num_scales, print_loss=False) for idx in range(num_scales)]

    train_dataset = strategy.experimental_distribute_dataset(train_dataset_org)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset_org)

    # tf.disable_eager_execution()
    with strategy.scope():
        factory = ModelFactory(tf.keras.layers.Input(shape=(*input_shape, 3)),
                               weights_path=model_path)

        # model_dummy = factory.build(mobilenetv2_yolo_body,
        #                       155,
        #                       0,
        #                       num_classes,
        #                       alpha=0.75,
        #                       model_wrapper=tf.keras.Model)
        # model_dummy.save('yolo_model.h5')
        # graph = tf.get_default_graph()
        # # print(graph.get_operations())
        # model_dummy.summary()
        # flops = tf.profiler.profile(tf.get_default_graph(),\
        #             options=tf.profiler.ProfileOptionBuilder.float_operation())
        # print('FLOP = ', flops.total_float_ops)
        # exit()

        if backbone == BACKBONE.MOBILENETV2x75:
            backbone_name = 'mobilenetv2x75'
        elif backbone == BACKBONE.MOBILENETV2x14:
            backbone_name = 'mobilenetv2x14'
        elif backbone == BACKBONE.EFFICIENTNETB3:
            backbone_name = 'efficientnetb3'

        model = factory.build(yolov3_body,
                              backbone_name,
                              len(anchors) // num_scales,
                              batch_norm_momentum=0.9,
                              batch_norm_epsilon=1e-3,
                              num_classes=num_classes,
                              drop_connect_rate=0.2,
                              data_format="channels_last")


    # exit()
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.

    if freeze is True:
        # for i in range(len(model.layers)):
        #     model.layers[i].trainable = True
        # model.load_weights(FLAGS['train_unfreeze'])
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[0],
                                                             epsilon=1e-8),
                          loss=loss)
        # model.save_weights(
        #     os.path.join(
        #         log_dir,
        #         str(backbone).split('.')[1].lower() +
        #         '_dummy_model.h5'))
        # exit()
        # tf.keras.utils.plot_model(model, to_file='model.png')
        # exit()
        # model.load_weights('logs/mobilenetv2_2020-04-09/ep042-loss13.528-val_loss13.472.h5')
        # exit()
        history_train, history_val = model.fit(epochs, [checkpoint, tensorboard, cos_lr_freeze], train_dataset, val_dataset)

        print(history_train)
        print(history_val)
        # plt.plot(history_train)
        # plt.plot(history_val)
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_stage_1.h5'))
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    else:
    # if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.load_weights(FLAGS['train_unfreeze'])
        with strategy.scope():
            model.compile(optimizer=tf.keras.optimizers.Adam(lr[1],
                                                             epsilon=1e-8),
                          loss=loss)  # recompile to apply the change
        print('Unfreeze all of the layers.')
        # history_train, history_val = model.fit(epochs, [checkpoint, tensorboard, early_stopping], train_dataset,
        #               val_dataset,use_adv=False)
        history_train, history_val = model.fit(epochs, [checkpoint, cos_lr, tensorboard], train_dataset,
                      val_dataset, use_adv=False)
        print(history_train)
        print(history_val)
        # plt.plot(history_train)
        # plt.plot(history_val)
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        model.save_weights(
            os.path.join(
                log_dir,
                str(backbone).split('.')[1].lower() +
                '_trained_weights_final.h5'))

    # Further training if needed.
