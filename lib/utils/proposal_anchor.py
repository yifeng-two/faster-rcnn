"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import tensorflow as tf
import numpy as np
from lib.utils.config import Config as cfg
from lib.utils.cython_bbox import bbox_overlaps
from lib.utils.bbox_transform import _transform_bboxes_targets, clip_boxes, _compute_targets


def _get_bbox_regression_labels(bbox_target_data, labels):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_classes = len(cfg.classes)
    clss = labels[:, np.newaxis]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 0:]
        bbox_inside_weights[ind, start:end] = cfg.train_bbox_inside_weights

    return bbox_targets, bbox_inside_weights


class ProposalCreator(object):
    """
    # 目的：为Fast-RCNN也即检测网络提供2000个训练样本
    # 输入：RPN网络中1*1卷积输出的loc和score，以及20000个anchor坐标，原图尺寸
    # 输出：2000个训练样本rois（只是2000*4的坐标，无ground truth！）
    """
    def __init__(self):

        self.rpn_nms_thresh = cfg.rpn_nms_thresh
        self.rpn_pre_nms_top_n = cfg.rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = cfg.rpn_post_nms_top_n

    def __call__(self, rpn_output_boxes_targets, rpn_fg_score, anchors, img_size):

        rpn_pred_boxes = _transform_bboxes_targets(anchors, rpn_output_boxes_targets)
        # 预测框设置为不大于图片大小
        rpn_pred_boxes = clip_boxes(rpn_pred_boxes, img_size)

        # 按照score大小排序
        inds = tf.argsort(rpn_fg_score, direction='DESCENDING').numpy()
        inds = inds.ravel()[:self.rpn_pre_nms_top_n]
        score_keep = rpn_fg_score[inds]
        rpn_pred_boxes = rpn_pred_boxes[inds, :]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).
        keep = tf.image.non_max_suppression(rpn_pred_boxes,
                                            score_keep,
                                            max_output_size=self.rpn_post_nms_top_n,
                                            iou_threshold=self.rpn_nms_thresh)
        rois = rpn_pred_boxes[keep.numpy()]

        return rois


class ProposalTargetCreator(object):
    """
    目的：为2000个rois赋予ground truth！（严格讲挑出128个赋予ground truth！）
    输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、
          对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
    输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）
    ProposalTargetCreator是RPN网络与ROIHead网络的过渡操作，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，
    而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：
        RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
        选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本
        # <取消了> 为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）
    对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.
    而且是只对正样本中的这个类别4个参数计算损失。举例来说:
    一个RoI在经过FC 84后会输出一个84维的loc向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss
    如果这个RoI是正样本, 属于label K, 那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失
    """
    def __init__(self):
        super(ProposalTargetCreator, self).__init__()
        self.rois_per_image = cfg.train_rois_per_image
        self.positive_fraction = cfg.train_positive_fraction

        self.negative_thresh_hi = cfg.train_negative_thresh_hi
        self.negative_thresh_lo = cfg.train_negative_thresh_lo
        self.positive_thresh = cfg.trian_positive_thresh
        self._num_classes = len(cfg.classes)

    def __call__(self, rois, gt_boxes, gt_labels):

        # n_gt_boxes, _ = gt_boxes.shape
        # 将gt_boxes 置于训练集中
        assert gt_boxes.shape[0] == 1, 'batchsize >1,make sure input on image at once'
        gt_boxes = gt_boxes.reshape((gt_boxes.shape[1], 4))
        assert gt_labels.shape[0] == 1, 'batchsize >1,make sure input on image at once'
        gt_labels = gt_labels[0]
        rois = np.concatenate((rois, gt_boxes), axis=0)
        train_poitive_rois_num = np.round(self.rois_per_image * self.positive_fraction)

        overlaps = bbox_overlaps(np.ascontiguousarray(rois[:, :4], dtype=np.float),
                                 np.ascontiguousarray(gt_boxes[:, :], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)

        # 设置rois对应的label
        roi_labels = gt_labels[gt_assignment]
        # 判断每个roi与gt_boxes 的iou最大值是否大于positive_thresh, 并且满足正例个数是否大于设置值
        positive_indexs = np.where(max_overlaps >= self.positive_thresh)[0]
        negative_indexs = np.where((max_overlaps >= self.negative_thresh_lo)
                                   & (max_overlaps < self.negative_thresh_hi))[0]
        if positive_indexs.size > 0 and negative_indexs.size > 0:
            positive_rois_this_image = int(min(train_poitive_rois_num, positive_indexs.size))
            positive_indexs = np.random.choice(positive_indexs,
                                               size=positive_rois_this_image,
                                               replace=False)
            negative_rois_this_image = self.rois_per_image - positive_rois_this_image
            to_replace = negative_indexs.size < negative_rois_this_image
            negative_indexs = np.random.choice(negative_indexs,
                                               size=negative_rois_this_image,
                                               replace=to_replace)
        elif positive_indexs.size > 0:
            to_replace = positive_indexs.size < self.rois_per_image
            positive_indexs = np.random.choice(positive_indexs,
                                               size=self.rois_per_image,
                                               replace=to_replace)
            positive_rois_this_image = self.rois_per_image
        elif negative_indexs.size > 0:
            to_replace = negative_indexs.size < self.rois_per_image
            negative_indexs = np.random.choice(negative_indexs,
                                               size=self.roi_per_image,
                                               replace=to_replace)
            positive_rois_this_image = 0
        else:
            import pdb
            pdb.set_trace()

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(positive_indexs, negative_indexs)
        roi_labels = roi_labels[keep_index]
        # negative labels --> 0
        roi_labels[positive_rois_this_image:] = 0
        roi_labels = np.array(roi_labels, dtype=np.int32)
        rois = rois[keep_index]

        roi_bbox_deltas = _compute_targets(rois, gt_boxes[gt_assignment[keep_index]])
        roi_bbox_targets, roi_bbox_inside_weights = _get_bbox_regression_labels(
            roi_bbox_deltas, roi_labels)
        roi_labels = roi_labels.reshape(-1, 1)
        roi_bbox_targets = roi_bbox_targets.reshape(-1, self._num_classes * 4)
        roi_bbox_inside_weights = roi_bbox_inside_weights.reshape(-1, self._num_classes * 4)
        roi_bbox_outside_weights = np.array(roi_bbox_inside_weights > 0).astype(np.float32)

        return rois, roi_labels, roi_bbox_targets, roi_bbox_inside_weights, roi_bbox_outside_weights
