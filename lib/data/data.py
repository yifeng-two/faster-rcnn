"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import numpy as np
import tensorflow as tf
from lib.utils.config import Config as cfg
from lib.data.pascal_voc import pascal_voc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DataSets(object):
    """
    长边不超过1000，短边不超过600，bbox坐标对应缩放。
    """
    def __init__(self, image_set):
        super(DataSets, self).__init__()
        self.data = pascal_voc(image_set)
        self.min_size = cfg.min_size
        self.max_size = cfg.max_size

    def _get_orig_img(self, img_path):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        return image

    def preprocess(self, image):
        image = image / 255.0
        # shape = image.shape.as_list()
        print(image.shape)
        H, W, C = image.shape
        img_size_min = min(H, W)
        img_size_max = max(H, W)
        # print(type(self.min_size), type(img_size_min))
        img_scale = self.min_size / img_size_min
        if np.round(img_scale * img_size_max) > self.max_size:
            img_scale = self.max_size / img_size_max
        image = tf.image.resize(image, [int(H * img_scale), int(W * img_scale)])
        image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])

        return image, img_scale

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        img_path = self.data.image_path_at(idx)
        # print(img_path)
        orig_image = self._get_orig_img(img_path)
        gt_boxes, gt_classes = self.data._load_pascal_annotation(idx)

        image, img_scale = self.preprocess(orig_image)
        gt_boxes = gt_boxes[:, 0:4] * img_scale

        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)

        return img_tensor, gt_boxes, gt_classes, img_scale


def process_data(image, height, width, bboxes):
    H = height.numpy()
    W = width.numpy()
    img_size_min = min(H, W)
    img_size_max = max(H, W)
    # print(type(self.min_size), type(img_size_min))
    img_scale = cfg.min_size / img_size_min
    if np.round(img_scale * img_size_max) > cfg.max_size:
        img_scale = cfg.max_size / img_size_max
    image = tf.image.resize(image, [int(H * img_scale), int(W * img_scale)])
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    # print(image.shape)
    # image = tf.expand_dims(image, axis=0)
    gt_boxes = bboxes[:, 0:4] * img_scale
    # print(bboxes.shape)
    bboxes = tf.cast(gt_boxes, dtype=tf.float32)
    return image, bboxes, img_scale


def create_dataset(filenames, batch_size=1, is_shuffle=False, n_repeats=0):
    """
    :param filenames: record file names
    :param batch_size:
    :param is_shuffle: 是否打乱数据
    :param n_repeats:
    :return:
    """
    files = tf.data.Dataset.list_files(filenames, shuffle=True)
    dataset = files.interleave(map_func=tf.data.TFRecordDataset,
                               cycle_length=len(filenames),
                               deterministic=False)
    # dataset = tf.data.TFRecordDataset(filenames)
    dataset_lens = len(list(dataset))
    dataset = dataset.map(lambda x: parse_single_example(x))

    if n_repeats > 0:
        dataset = dataset.repeat(n_repeats)  # for train
    if n_repeats == -1:
        dataset = dataset.repeat()  # for val to

    if is_shuffle:
        dataset = dataset.shuffle(2)  # shuffle
    dataset = dataset.batch(batch_size)
    # 提高性能
    dataset.prefetch(batch_size)

    return dataset, dataset_lens


def parse_single_example(serialized_example):
    """
    解析tf.record
    :param serialized_example:
    :return:
    """
    feature_dict = tf.io.parse_single_example(serialized_example,
                                              features={
                                                  'image/height':
                                                  tf.io.FixedLenFeature([], dtype=tf.int64),
                                                  'image/width':
                                                  tf.io.FixedLenFeature([], dtype=tf.int64),
                                                  'image/channels':
                                                  tf.io.FixedLenFeature([], dtype=tf.int64),
                                                  'image/filename':
                                                  tf.io.FixedLenFeature([], dtype=tf.string),
                                                  'image/image_id':
                                                  tf.io.FixedLenFeature([], dtype=tf.string),
                                                  'image/object/bbox/xmin':
                                                  tf.io.VarLenFeature(dtype=tf.float32),
                                                  'image/object/bbox/xmax':
                                                  tf.io.VarLenFeature(dtype=tf.float32),
                                                  'image/object/bbox/ymin':
                                                  tf.io.VarLenFeature(dtype=tf.float32),
                                                  'image/object/bbox/ymax':
                                                  tf.io.VarLenFeature(dtype=tf.float32),
                                                  'image/object/bbox/label_text':
                                                  tf.io.VarLenFeature(dtype=tf.string),
                                                  'image/object/bbox/label':
                                                  tf.io.VarLenFeature(dtype=tf.int64),
                                                  'image/object/bbox/difficult':
                                                  tf.io.VarLenFeature(dtype=tf.int64),
                                                  'image/object/bbox/truncated':
                                                  tf.io.VarLenFeature(dtype=tf.int64),
                                                  'image/format':
                                                  tf.io.FixedLenFeature([], dtype=tf.string),
                                                  'image/encoded':
                                                  tf.io.FixedLenFeature([], dtype=tf.string),
                                              })
    # shape = feature_dict['image/shape']
    image_id = feature_dict['image/image_id']
    height = tf.cast(feature_dict['image/height'], tf.int32)
    width = tf.cast(feature_dict['image/width'], tf.int32)
    # channels = feature_dict['image/channels']
    image = tf.io.decode_jpeg(feature_dict['image/encoded'], channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.0
    bboxes = tf.stack([
        tf.sparse.to_dense(feature_dict['image/object/bbox/xmin']),
        tf.sparse.to_dense(feature_dict['image/object/bbox/ymin']),
        tf.sparse.to_dense(feature_dict['image/object/bbox/xmax']),
        tf.sparse.to_dense(feature_dict['image/object/bbox/ymax'])
    ],
                      axis=1)
    labels = tf.cast(tf.sparse.to_dense(feature_dict['image/object/bbox/label']), tf.float32)

    # image, bboxes = process_data(image, H, W, bboxes)
    # print(image, height, width, channels, bboxes, labels)
    return image_id, image, bboxes, labels, height, width


if __name__ == '__main__':
    training_dataset = tf.data.TFRecordDataset("")
    plt.figure(figsize=(15, 10))
    i = 0
    for image, boxes, boxes_category in training_dataset.take(12):
        plt.subplot(3, 4, i + 1)

        plt.imshow(image)
        ax = plt.gca()
        for j in range(boxes.shape[0]):
            rect = Rectangle((boxes[j, 0], boxes[j, 1]),
                             boxes[j, 2] - boxes[j, 0],
                             boxes[j, 3] - boxes[j, 1],
                             color='r',
                             fill=False)
            ax.add_patch(rect)
        i += 1
    plt.show()
