"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import datetime
import argparse
import sys, os
import tensorflow as tf
from lib.utils.config import Config as cfg
from lib.data.data import process_data, create_dataset
from lib.data.create_pascal_tf_record import generate_tfrecord
from lib.nets.fasterrcnn import FasterRCNN
from lib.backbones.vgg16 import VGG16
from lib.backbones.resnet import ResNetV1
from lib.backbones.mobilenet import MobileNetV1
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--epoch', dest='max_epochs', help='optional epoch', default=20, type=int)
    parser.add_argument('--iters',
                        dest='max_iters',
                        help='number of iterations to train',
                        default=70000,
                        type=int)
    parser.add_argument('--lr',
                        dest='learning_rate',
                        help='optional learning_rate',
                        default=1e-4,
                        type=float)
    parser.add_argument('--mt', dest='momentum', help='optional momentum', default=0.9, type=float)
    parser.add_argument('--batch',
                        dest='batch_size',
                        help='optional batch size',
                        default=1,
                        type=int)
    parser.add_argument(
        '--weight',
        dest='weight',
        help='initialize with pretrained model weights,only need model name like vgg16.ckpt',
        default="vgg16.ckpt",
        type=str)
    parser.add_argument('--pretrianed',
                        dest='use_pretrained',
                        help='whether initialize with pretrained model weights',
                        default=False,
                        type=bool)
    parser.add_argument('--log',
                        dest='log_path',
                        help='logging for save train',
                        default="./logs/",
                        type=str)
    parser.add_argument('--tag', dest='tag', help='tag of the model', default=None, type=str)
    parser.add_argument('--net',
                        dest='net',
                        help='vgg16, res18, res34,res50, res101, res152, mobilenetV1',
                        default='vgg16',
                        type=str)
    parser.add_argument('--set',
                        dest='set_cfgs',
                        help='set config keys',
                        default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def train(args, dataset, net):

    model = FasterRCNN(net)
    # load pre_train_weights
    if not tf.io.gfile.exists(cfg.pretrained_weights):
        tf.io.gfile.makedirs(cfg.pretrained_weights)
    if args.use_pretrained:
        pretrained_weights_path = os.path.join(cfg.pretrained_weights, args.weight)
        if not tf.io.gfile.exists(pretrained_weights_path):
            print('no initial model weights file in {:s}'.format(cfg.pretrained_weights))
        weights_suffix = os.path.splitext(pretrained_weights_path)[-1]
        print('Loading initial model weights from {:s}'.format(pretrained_weights_path))
        if weights_suffix == '.ckpt':
            model.load_weights(pretrained_weights_path, by_name=False)
        if weights_suffix == '.h5':
            model.load_weights(pretrained_weights_path, by_name=True)

    # print(args)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.log_path + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    # with summary_writer.as_default():
    #     tf.summary.graph(model.get_concrete_function().graph)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum)

    lens = dataset[1]
    train_iterator = iter(dataset[0])

    # print(len(list(dataset)), lens)
    for epoch in range(args.max_epochs):
        for i in range(lens):
            image_id, image, bboxes, labels, height, width = train_iterator.get_next()
            image, bboxes, _ = process_data(image, height, width, bboxes)
            model._get_data_input(image, bboxes, labels)
            with tf.GradientTape() as tape:
                rpn_cls_loss, rpn_pred_boxes_loss, roi_cls_loss, roi_pred_bboxes_loss = model(
                    image, training=True)
                total_loss = rpn_cls_loss + rpn_pred_boxes_loss + roi_cls_loss + roi_pred_bboxes_loss
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if i % 1 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('rpn_cls_loss', float(rpn_cls_loss), step=i + epoch * lens)
                    tf.summary.scalar('rpn_pred_boxes_loss',
                                      float(rpn_pred_boxes_loss),
                                      step=i + epoch * lens)
                    tf.summary.scalar('roi_cls_loss', float(roi_cls_loss), step=i + epoch * lens)
                    tf.summary.scalar('roi_pred_bboxes_loss',
                                      float(roi_pred_bboxes_loss),
                                      step=i + epoch * lens)
            if (i + epoch * lens) % 20 == 0:
                print('trianing step =', i + epoch * lens, 'rpn_cls_loss=', rpn_cls_loss.numpy(),
                      'rpn_pred_boxes_loss=', rpn_pred_boxes_loss.numpy(), 'roi_cls_loss=',
                      roi_cls_loss.numpy(), 'roi_pred_bboxes_loss=', roi_pred_bboxes_loss.numpy(),
                      'total_loss=', total_loss.numpy())
            if (i + epoch * lens) % 100 == 0 and (i + epoch * lens) != 0:
                # if (i + epoch * lens) % 20 == 0:
                # # 对于自定义模型，给模型制定一个输入形状，这对于后面模型的保存以及加载，是有必要的
                # # 通过TensorSpec创建一个“无实际数据的张量”，指定它的形状，作为模型的输入
                # input_shape = tf.TensorSpec(shape=(1, 1000, 600, 3),
                #                             dtype=tf.dtypes.float32,
                #                             name=None)
                # # 设置模型的输入
                # model._set_inputs(input_shape)  # 设置模型的输入形状

                # # save_weights 仅保存模型的权重和偏置
                model.save_weights(log_dir + '/frcnn.ckpt')
                # save将网络的结构，权重和优化器的状态等参数全部保存下来
                # # tf.saved_model.save(model, log_dir)

                # checkpoint = tf.train.Checkpoint(myModel=model)
                # checkpoint.save(log_dir + "/frcnn_{:s}.ckpt".format(str(i + epoch * lens)))


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(devices[0], True)

    args = parse_args()

    print('Called with args:')
    print(args)

    # 输出configuration
    config = cfg()
    config.display()

    # print("start loading datasets......")
    # dataset = DataSets(config.image_set)
    # print("datasets has %s images" % (len(dataset)))
    # 设置数据集

    if (tf.io.gfile.exists(cfg.output_dir)):
        if tf.io.gfile.listdir(cfg.output_dir):
            print("datasets already transform to tfrecord files,please checkout {:s}".format(
                cfg.output_dir))
    else:
        for split in cfg.split_map:
            generate_tfrecord(cfg.voc_root, 2007, split, output_dir=cfg.output_dir)
    # 读取tfrecord文件并列成列表，train_dir是存放的路径
    train_dir = os.path.join(cfg.output_dir, "%s" % cfg.split_map[0])
    val_dir = os.path.join(cfg.output_dir, "%s" % cfg.split_map[1])
    # test_dir = os.path.join(cfg.output_dir, "%s" % cfg.split_map[2])

    train_file_names = [os.path.join(train_dir, i) for i in os.listdir(train_dir)]
    val_file_names = [os.path.join(val_dir, i) for i in os.listdir(val_dir)]
    # 定义数据集
    # # train_dataset 用epochs控制循环

    training_dataset = create_dataset(train_file_names,
                                      batch_size=args.batch_size,
                                      is_shuffle=True,
                                      n_repeats=args.max_epochs)  # train_filename

    # load network
    if args.net == 'vgg16':
        net = VGG16()
    elif args.net == 'res18':
        net = ResNetV1(num_layers=18)
    elif args.net == 'res34':
        net = ResNetV1(num_layers=34)
    elif args.net == 'res50':
        net = ResNetV1(num_layers=50)
    elif args.net == 'res101':
        net = ResNetV1(num_layers=101)
    elif args.net == 'res152':
        net = ResNetV1(num_layers=152)
    elif args.net == 'mobilenetV1':
        net = MobileNetV1()
    else:
        raise NotImplementedError

    train(args, training_dataset, net)
