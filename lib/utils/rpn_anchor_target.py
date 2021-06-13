"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import numpy as np
from lib.utils.config import Config as config
from lib.utils.cython_bbox import bbox_overlaps
from lib.utils.bbox_transform import _compute_targets


def _generate_base_anchor(anchor_base_size=16,
                          ratios=[0.5, 1, 2],
                          anchor_scales=[8, 16, 32]):
    """
    这个函数的作用就是产生(0,0)坐标开始的基础的9个anchor框，再根据不同的放缩比和宽高比进行进一步的调整。
    本代码中对应着三种面积的大小(16*8)^2 (16*16)^2 (16*32)^2
    也就是128,256,512的平方大小，三种面积乘以三种放缩比就刚刚好是9种anchor
    :param anchor_base_size: 基础的anchor的宽和高是16的大小(stride),对应原图16*16区域大小
    :param ratios: 宽高的比例
    :param anchor_scales: 在base_size的基础上再增加的量
    :return: |ratios| * |scales|个anchors的坐标
    """
    # anchor 中心点
    ctr_x = anchor_base_size / 2
    ctr_y = anchor_base_size / 2

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    # 生成九种尺度的anchor
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = anchor_base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = anchor_base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = ctr_x - w / 2.
            anchor_base[index, 1] = ctr_y - h / 2.
            anchor_base[index, 2] = ctr_x + w / 2.
            anchor_base[index, 3] = ctr_y + h / 2.

    return anchor_base


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    根据上面函数在原图的第(0,0)个特征图感受野中心位置生成的anchor_base, 在原图的特征图感受野中心生成anchors
    :param anchor_base: anchors的坐标
    :param feat_stride: 特征图缩小倍数（步长）
    :param height: 特征图高度
    :param width: 特征图宽度
    :return: 原图上所有的anchors坐标，[h * w * #anchor,4]
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # print(shift_x.shape,shift_y.shape)
    shift = np.stack(
        (shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()),
        axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    # print(anchor_base.reshape((1, A, 4)))
    # print(shift.reshape((1, K, 4)).transpose((1, 0, 2)))
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape(
        (1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of
    size count)
    """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


class AnchorTargetCreator(object):
    """
    目的：利用每张图中bbox的真实标签来为所有任务分配ground truth！
    输入：最初生成的20000个anchor坐标、此一张图中所有的bbox的真实坐标
    输出：size为（20000，1）的正负label（其中只有128个为1，128个为0，其余都为-1
          size为（20000，4）的回归目标（所有anchor的坐标都有）
    将20000多个候选的anchor选出256个anchor进行二分类和所有的anchor进行回归位置 。为上面的预测值提供相应的真实值。选择方式如下：
    对于每一个ground truth bounding box (gt_bbox)，选择和它重叠度（IoU）最高的一个anchor作为正样本。
    对于剩下的anchor，从中选择和任意一个gt_bbox重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
    随机选择和gt_bbox重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。
    对于每个anchor, gt_label 要么为1（前景），要么为0（背景），所以这样实现二分类。
    在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。
    """
    def __init__(self):
        super(AnchorTargetCreator, self).__init__()
        # rpn_train_batch=256,设置一张图训练正样本为一半，正样本不足时，用负样本填充
        self.rpn_trian_batch_size = config.rpn_train_batch_size
        self.rpn_positive_overlap = config.rpn_positive_overlap
        self.rpn_negative_overlap = config.rpn_negative_overlap
        self.rpn_positive_ratio = config.rpn_positive_ratio

    def __call__(self, bbox, anchors, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchors)
        # 获取在图片范围内的anchor的index值
        inside_index = np.where((anchors[:, 0] >= 0) &  # y1
                                (anchors[:, 1] >= 0) &  # x1
                                (anchors[:, 2] <= img_H) &  # y2
                                (anchors[:, 3] <= img_W)  # x2)
                                )[0]
        # keep only inside anchors
        anchors = anchors[inside_index]
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self._create_label(
            n_anchor, inside_index, anchors, bbox)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def _create_label(self, n_anchor, inside_index, anchors, gt_boxes):

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inside_index), ), dtype=np.int32)
        labels.fill(-1)

        assert gt_boxes.shape[0] == 1, 'batchsize >1,make sure input on image at once'
        gt_boxes = gt_boxes.reshape((gt_boxes.shape[1], 4))
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        # argmax_overlaps每个anchor对应iou最大的gt_box的index
        # max_overlaps为每个anchor对应最大gt_box的iou值
        # gt_argmax_overlaps 为对于每一个(gt_bbox)，和它IoU最高的一个anchor的index位置
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inside_index)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # 为anchor分配label
        # 每个anchor对应最大gt_box的iou值小于0.3时，设置label为背景,label=0
        # 对应最大gt_box的iou值小于0.3时，设置label为前景,label=1
        labels[max_overlaps >= self.rpn_positive_overlap] = 1
        labels[max_overlaps < self.rpn_negative_overlap] = 0
        # 对于每一个(gt_bbox)，和它IoU最高的一个anchor的设置为前景,label =1
        labels[gt_argmax_overlaps] = 1

        # 对已生成的label进行前景和背景训练数量进行调整
        # 若前景数量超出设置比例，则随机选择设置为label = -1(即忽略)
        positive_samples_num = int(self.rpn_trian_batch_size *
                                   self.rpn_positive_ratio)
        positive_samples_inds = np.where(labels == 1)[0]
        if (len(positive_samples_inds) > positive_samples_num):
            disable_inds = np.random.choice(positive_samples_inds,
                                            size=(len(positive_samples_inds) -
                                                  positive_samples_num),
                                            replace=False)
            labels[disable_inds] = -1
        # 若背景数量超出设置比例，则随机选择设置为label = -1(即忽略)
        negative_samples_num = self.rpn_trian_batch_size - np.sum(labels == 1)
        negative_samples_inds = np.where(labels == 0)[0]
        if (len(negative_samples_inds) > negative_samples_num):
            disable_inds = np.random.choice(negative_samples_inds,
                                            size=(len(negative_samples_inds) -
                                                  negative_samples_num),
                                            replace=False)
            labels[disable_inds] = -1

        # 计算anchor 与gt_boxes 之间的偏移量
        rpn_bbox_targets = np.zeros((len(inside_index), 4), dtype=np.float32)
        rpn_bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        rpn_bbox_inside_weights = np.zeros((len(inside_index), 4),
                                           dtype=np.float32)
        # only the positive ones have regression targets
        rpn_bbox_inside_weights[labels == 1, :] = np.array(
            config.train_rpn_bbox_inside_weights)

        rpn_bbox_outside_weights = np.zeros((len(inside_index), 4),
                                            dtype=np.float32)
        if config.train_rpn_positive_weight < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((config.train_rpn_positive_weight > 0) &
                    (config.train_rpn_positive_weight < 1))
            positive_weights = (config.train_rpn_positive_weight /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - config.train_rpn_positive_weight) /
                                np.sum(labels == 0))
        rpn_bbox_outside_weights[labels == 1, :] = positive_weights
        rpn_bbox_outside_weights[labels == 0, :] = negative_weights

        # 将训练数据返回至所有生成的anchor中
        labels = _unmap(labels, n_anchor, inside_index, fill=0)
        rpn_bbox_targets = _unmap(rpn_bbox_targets,
                                  n_anchor,
                                  inside_index,
                                  fill=0)
        rpn_bbox_inside_weights = _unmap(rpn_bbox_inside_weights,
                                         n_anchor,
                                         inside_index,
                                         fill=0)
        rpn_bbox_outside_weights = _unmap(rpn_bbox_outside_weights,
                                          n_anchor,
                                          inside_index,
                                          fill=0)

        return labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


if __name__ == "__main__":
    anchors_base = _generate_base_anchor()
    anchors = _enumerate_shifted_anchor(np.array(anchors_base), 16, 38, 50)
    # print(anchors)