"""
Faster R-CNN
Base Configurations class.

Copyright (c) 2020 yifeng, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.


class Config(object):

    # data param configuration
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    voc_root = os.path.join(
        root_dir,
        "datasets",
        "VOCdekit",
    )
    data_dir = os.path.join(root_dir, "datasets", "VOCdekit", "VOC2007")
    pretrained_weights = os.path.join(root_dir, "imagenet_weights")
    output_dir = os.path.join(root_dir, "datasets", 'tfrecords', 'VOCdekit', "VOC2007")

    # classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')
    classes = ('__background__', 'car')
    image_set = "trainval.txt"
    split_map = ['train', 'val', 'test']
    # image input setting
    max_size = 1000
    min_size = 600
    # anchor param
    feat_stride = 16
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]

    # rpn network configuration
    train_rpn_bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)
    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting
    train_rpn_positive_weight = -1.0

    rpn_train_batch_size = 128
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_positive_ratio = 0.5

    # some nms thresh configuration
    rpn_nms_thresh = 0.7
    rpn_pre_nms_top_n = 12000
    rpn_post_nms_top_n = 2000

    # train fast rcnn layer configuration
    train_negative_thresh_hi = 0.5
    train_negative_thresh_lo = 0.3
    trian_positive_thresh = 0.7
    train_rois_per_image = 128
    train_positive_fraction = 0.25

    # ROI Pooling configuration
    pool_size = (7, 7)

    # losses configuration
    train_bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)

    # detect configration
    score_thresh = 0.7
    nms_thresh = 0.3

    test_thresh = 0.5
    test_max_per_image = 300

    use_salt = True
    clean_up = True
    use_diff = False

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for value in dir(self):
            if not value.startswith("__") and not callable(getattr(self, value)):
                print("{:30} {}".format(value, getattr(self, value)))
        print("\n")


if __name__ == "__main__":
    config = Config()
    config.display()