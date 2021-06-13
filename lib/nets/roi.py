"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.

"""

import tensorflow as tf
from lib.utils.config import Config as cfg


class ROIPooling2D(tf.keras.Model):
    def __init__(self, pool_size):
        super(ROIPooling2D, self).__init__()
        self.pool_size = pool_size

    def call(self, fearure, rois, img_size):
        return self.roi_pooling(fearure, rois, img_size, self.pool_size)

    def roi_pooling(self, feature, rois, img_size, pool_size):
        '''
        Regions of Interest (ROIs) from the Region Proposal Network (RPN) are
        formatted as:
        (image_id, x1, y1, x2, y2)
        Note: Since mini-batches are sampled from a single image, image_id = 0s
        '''
        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
        box_ind = tf.zeros(rois.shape[0], dtype=tf.int32)
        # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
        normalization = tf.cast(tf.stack([img_size[1], img_size[0], img_size[1], img_size[0]],
                                         axis=0),
                                dtype=tf.float32)
        boxes = rois / normalization
        boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
        # ROI pool output size
        # crop_size = tf.constant([14, 14])
        crop_size = [i * 2 for i in pool_size]
        # ROI pool
        pooledFeatures = tf.image.crop_and_resize(image=feature,
                                                  boxes=boxes,
                                                  box_indices=box_ind,
                                                  crop_size=crop_size)
        # Max pool to (7x7)
        pooledFeatures = tf.nn.max_pool(pooledFeatures,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME')

        return pooledFeatures


class ROIHead(tf.keras.Model):
    def __init__(self):
        super(ROIHead, self).__init__()
        self.n_classes = len(cfg.classes)
        self.pool_size = cfg.pool_size
        self.ROIPooling2D = ROIPooling2D(self.pool_size)

        self.fc = tf.keras.layers.Dense(4096)
        self.pred_cls = tf.keras.layers.Dense(self.n_classes)
        self.pred_boxes = tf.keras.layers.Dense(self.n_classes * 4)

    def call(self, feature, rois, img_size, training=None):

        rois = tf.constant(rois, dtype=tf.float32)
        pool = self.ROIPooling2D(feature, rois, img_size)
        pool = tf.reshape(pool, [rois.shape[0], -1])
        fc = self.fc(pool)
        roi_pred_boxes = self.pred_boxes(fc)
        roi_scores = self.pred_cls(fc)

        return roi_scores, roi_pred_boxes
