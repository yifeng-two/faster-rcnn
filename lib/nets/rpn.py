"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import tensorflow as tf
from lib.utils.config import Config as cfg
from lib.utils.rpn_anchor_target import _generate_base_anchor, _enumerate_shifted_anchor
from lib.utils.proposal_anchor import ProposalCreator


class RegionProposalNet(tf.keras.Model):
    def __init__(self):
        super(RegionProposalNet, self).__init__()
        self.region_proposal_conv = tf.keras.layers.Conv2D(512,
                                                           kernel_size=3,
                                                           padding='same',
                                                           activation='relu')
        self.rpn_boxes_pred = tf.keras.layers.Conv2D(36, kernel_size=3, padding='same')
        self.rpn_cls_score = tf.keras.layers.Conv2D(18, kernel_size=1, padding='same')
        self.anchor_base = _generate_base_anchor(anchor_scales=cfg.anchor_scales, ratios=cfg.ratios)
        self.proposal_layer = ProposalCreator()

    def call(self, x, img_size, training=None):
        # print(x.shape)
        n, h, w, _ = x.shape
        # n, h, w, _ = tf.shape(x)
        # n = 1
        anchor = _enumerate_shifted_anchor(self.anchor_base, cfg.feat_stride, h, w)
        n_anchor = anchor.shape[0] // (h * w)
        x = self.region_proposal_conv(x)
        # rpn_boxes_pred [1, 38, 50, 36]
        rpn_boxes_pred = self.rpn_boxes_pred(x)
        rpn_boxes_pred = tf.reshape(rpn_boxes_pred, [n, -1, 4])
        # rpn_cls_sore [1, 38, 50, 18]
        rpn_cls_score = self.rpn_cls_score(x)
        # 应用softmax函数  # [1, 38, 50, 9, 2]
        rpn_softmax_score = tf.nn.softmax(tf.reshape(rpn_cls_score, [n, h, w, n_anchor, 2]),
                                          axis=-1)
        rpn_fg_score = rpn_softmax_score[:, :, :, :, 1]
        rpn_fg_score = tf.reshape(rpn_fg_score, [n, -1])
        # rpn_cls_score reshape
        rpn_cls_score = tf.reshape(rpn_cls_score, [n, -1, 2])

        rois = self.proposal_layer(rpn_boxes_pred[0].numpy(), rpn_fg_score[0].numpy(), anchor,
                                   img_size)

        return rpn_cls_score, rpn_boxes_pred, rois, anchor
