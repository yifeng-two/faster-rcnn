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
from lib.nets.fasterrcnn import FasterRCNN
from lib.backbones.vgg16 import VGG16
from lib.backbones.resnet import ResNetV1
from lib.backbones.mobilenet import MobileNetV1
import matplotlib.pyplot as plt
import cv2


def _get_orig_img(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    return image


def preprocess(image):
    image = image / 255.0
    H, W, C = image.shape
    img_size_min = min(H, W)
    img_size_max = max(H, W)
    img_scale = float(config.min_size) / float(img_size_min)
    if np.round(img_scale * img_size_max) > config.max_size:
        img_scale = float(config.max_size) / float(img_size_max)
    image = tf.image.resize(image, [int(H * img_scale), int(W * img_scale)])
    image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])

    return image, img_scale


def process_output(boxes, scores):
    out_bboxes = []
    out_labels = []
    out_scores = []
    for j in range(1, len(cfg.classes)):
        inds = np.where(scores[:, j] > 0.5)[0]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_scores = scores[inds, j]

        keep = tf.image.non_max_suppression(cls_boxes,
                                            cls_scores,
                                            max_output_size=-1,
                                            iou_threshold=cfg.nms_thresh)
        if len(keep) > 0:
            out_bboxes.append(cls_boxes[keep.numpy()])
            # The labels are in [1, self.n_class].
            out_labels.append((j) * np.ones((len(keep), )))
            out_scores.append(cls_scores[keep.numpy()])
    if len(out_bboxes) > 0:
        out_bboxes = np.concatenate(out_bboxes, axis=0).astype(np.float32)
        out_labels = np.concatenate(out_labels, axis=0).astype(np.float32)
        out_scores = np.concatenate(out_scores, axis=0).astype(np.float32)
    return out_bboxes, out_labels, out_scores


def vis_detection(im_path, bboxes, scores, thresh=0.5):

    bboxes, labels, scores = process_output(boxes, scores)

    im = cv2.imread(img_path)
    # im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    # print(scores)
    inds = np.where(scores >= 0.5)[0]
    if (len(inds) == 0):
        return
    for i in inds:
        bbox = bboxes[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False,
                          edgecolor='red',
                          linewidth=3.5))
        ax.text(bbox[0],
                bbox[1] - 2,
                '{:s} {:.3f}'.format(cfg.classes[int(labels[i])], scores[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14,
                color='white')
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()


if __name__ == '__main__':
    devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(devices[0], True)

    # 输出configuration
    config = cfg()
    config.display()

    img_path = './demo/000456.jpg'
    orig_image = _get_orig_img(img_path)
    image, im_scale = preprocess(orig_image)
    img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    net = VGG16()
    model = FasterRCNN(net)

    model.load_weights('./logs/20210612-114230/frcnn.ckpt')
    boxes, scores = model.predict(img_tensor, im_scale)
    # print(bboxes, labels, scores)

    vis_detection(img_path, boxes, scores)
