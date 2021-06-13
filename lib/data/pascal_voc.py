# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import uuid
import numpy as np
import xml.etree.ElementTree as ET
from lib.utils.config import Config as config


class pascal_voc(object):
    def __init__(self, image_set, use_diff=False):
        super(pascal_voc, self).__init__()
        self._data_path = config.data_dir
        self.classes = config.classes
        self.image_set = config.image_set
        self._class_to_ind = dict(
            list(zip(self.classes, list(range(len(self.classes))))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.configs = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': use_diff,
            'matlab_eval': False,
            'rpn_file': None
        }

        assert os.path.exists(self._data_path), 'VOCdevkit path does not exist: {}'.format(self._data_path)

    def __len__(self):
        return len(self._image_index)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets/Main',
                                      self.image_set)
        assert os.path.exists(image_set_file), 'Path does not exist:{}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        f.close()
        return image_index

    def _load_pascal_annotation(self, index):
        filename = os.path.join(self._data_path, 'Annotations',
                                self._image_index[index] + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.configs['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0
            ]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # overlaps = np.zeros((num_objs,self.classes),dtype=np.float32)
        # seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            # overlaps[ix, cls] = 1.0
            # seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # overlaps = scipy.sparse.csr_matrix(overlaps)

        # return {'boxes': boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps': overlaps,
        #         'flipped': False,
        #         'seg_areas': seg_areas}
        return boxes, gt_classes
