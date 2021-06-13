
import numpy as np
import tensorflow as tf


def _compute_targets(ex_rois, gt_rois):
    """
    计算以x,y,w,h形式的两个bbox之间的offset.
    :param ex_rois: [?, 4]可以是anchors
    :param gt_rois: [?, 4]可以是ground truth
    :return: [?, 4] 对应每个anchors变换到gt的dx,dy,dh,dw四个参数
    """
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    # ex_rois 的H,W,ctr_x,ctr_y
    ex_height = ex_rois[:, 3] - ex_rois[:, 1] + 1
    ex_width = ex_rois[:, 2] - ex_rois[:, 0] + 1
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_width
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_height
    # gt_rois 的H,W,ctr_x,ctr_y
    gt_height = gt_rois[:, 3] - gt_rois[:, 1] + 1
    gt_width = gt_rois[:, 2] - gt_rois[:, 0] + 1
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_height

    # 计算出target 参数，也就是中心点和宽、高偏移量
    target_dw = np.log(gt_width / ex_width)
    target_dh = np.log(gt_height / ex_height)
    target_dx = (gt_ctr_x - ex_ctr_x) / ex_width
    target_dy = (gt_ctr_y - ex_ctr_y) / ex_height

    targets = np.vstack((target_dx, target_dy, target_dw, target_dh)).transpose()
    return targets


def _transform_bboxes_targets(boxes, deltas):

    if boxes.shape == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    # 生成的anchor
    height = boxes[:, 3] - boxes[:, 1] + 1
    width = boxes[:, 2] - boxes[:, 0] + 1
    ctr_x = boxes[:, 2] - width * 0.5
    ctr_y = boxes[:, 3] - height * 0.5

    # rpn 输出的dx,dy,dw,dh
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # 变换为x1,y1,x2,y2格式bboxes
    pred_ctr_x = dx * width[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * height[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * width[:, np.newaxis]
    pred_h = np.exp(dh) * height[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
  Clip boxes to image boundaries.
  """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def clip_boxes_tf(boxes, img_size):
    b0 = tf.maximum(tf.minimum(boxes[:, 0], img_size[1] - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], img_size[0] - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], img_size[1] - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], img_size[0] - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)
