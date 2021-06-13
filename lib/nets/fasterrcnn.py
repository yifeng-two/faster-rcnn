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
from lib.nets.rpn import RegionProposalNet
from lib.nets.roi import ROIHead
from lib.utils.rpn_anchor_target import AnchorTargetCreator
from lib.utils.proposal_anchor import ProposalTargetCreator
from lib.utils.bbox_transform import _transform_bboxes_targets, clip_boxes


def _smooth_l1_loss(pred_boxes, gt_boxes, bbox_inside_weights, bbox_outside_weights, sigma=1.0):
    """
    :param pred_boxes: 1,38,50,36
    :param gt_boxes: 17100,4
    :param gt_labels: 17100
    """
    sigma_2 = sigma**2
    box_diff = pred_boxes - gt_boxes
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    # smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    smoothL1_sign = tf.cast(abs_in_box_diff.numpy() < (1. / sigma_2), dtype=tf.float32)
    in_loss_box = tf.pow(
        in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff -
                                                            (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
    return loss_box


class FasterRCNN(tf.keras.Model):
    def __init__(self, extractor):
        super(FasterRCNN, self).__init__()
        self.classes = len(cfg.classes)
        self.extractor = extractor
        self.rpn = RegionProposalNet()
        self.rpn_anchor_target_creator = AnchorTargetCreator()
        self.proposal_anchor_target_creator = ProposalTargetCreator()
        self.ROI = ROIHead()

    def _get_data_input(self, image, gt_boxes, gt_labels):
        _, H, W, _ = image.shape
        self.img_size = (H, W)
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels

    def call(self, x, training=None):

        feature = self.extractor(x, training=training)
        rpn_cls_score, rpn_boxes_pred, rois, anchors = self.rpn(feature,
                                                                self.img_size,
                                                                training=training)
        rpn_score = rpn_cls_score[0]
        rpn_boxes = rpn_boxes_pred[0]

        sample_rois, roi_labels, roi_bbox_targets, roi_bbox_inside_weights, roi_bbox_outside_weights = self.proposal_anchor_target_creator(
            rois, self.gt_boxes.numpy(), self.gt_labels.numpy())
        roi_scores, roi_pred_boxes = self.ROI(feature, sample_rois, self.img_size)

        # RPN Loss
        rpn_gt_labels, rpn_gt_boxes, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.rpn_anchor_target_creator(
            self.gt_boxes.numpy(), anchors, self.img_size)
        rpn_gt_boxes = tf.constant(rpn_gt_boxes, dtype=tf.float32)
        rpn_gt_labels = tf.constant(rpn_gt_labels, dtype=tf.float32)
        rpn_pred_boxes_loss = _smooth_l1_loss(rpn_boxes,
                                              rpn_gt_boxes,
                                              rpn_bbox_inside_weights,
                                              rpn_bbox_outside_weights,
                                              sigma=3.0)
        idx = rpn_gt_labels != -1
        rpn_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
            rpn_gt_labels[idx], rpn_score[idx])

        # RCNN Loss
        roi_labels = tf.constant(roi_labels, dtype=tf.float32)
        roi_bbox_targets = tf.constant(roi_bbox_targets, dtype=tf.float32)
        roi_pred_bboxes_loss = _smooth_l1_loss(roi_pred_boxes,
                                               roi_bbox_targets,
                                               roi_bbox_inside_weights,
                                               roi_bbox_outside_weights,
                                               sigma=1.0)
        idx = roi_labels != 0
        roi_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(roi_labels,
                                                                                        roi_scores)

        return rpn_cls_loss, rpn_pred_boxes_loss, roi_cls_loss, roi_pred_bboxes_loss

    def predict(self, imgs, im_scale):

        img_size = imgs.shape[1:3]
        feature_map = self.extractor(imgs)
        rpn_cls_score, rpn_boxes_pred, rois, anchor = self.rpn(feature_map, img_size)
        roi_scores, roi_pred_boxes = self.ROI(feature_map, rois, img_size)

        roi_cls_scores = tf.nn.softmax(roi_scores, axis=-1)
        roi_cls_scores = roi_cls_scores.numpy()
        roi_pred_boxes = roi_pred_boxes.numpy()
        # roi_pred_boxes = roi_pred_boxes.reshape(-1, self.classes, 4)  # 2000, 21, 4

        boxes = rois[:, 1:5] / im_scale
        scores = np.reshape(roi_cls_scores, [roi_cls_scores.shape[0], -1])
        pre_boxes = np.reshape(roi_pred_boxes, [roi_pred_boxes.shape[0], -1])

        boxes = _transform_bboxes_targets(rois, pre_boxes)
        # clip bounding box
        boxes = clip_boxes(boxes, img_size)

        return boxes, scores
