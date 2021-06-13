# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import argparse
import uuid
import numpy as np
import tensorflow as tf
from lib.utils.timer import Timer
from lib.data.data import process_data, create_dataset
from lib.utils.config import Config as cfg
from lib.backbones.vgg16 import VGG16
from lib.backbones.resnet import ResNetV1
from lib.backbones.mobilenet import MobileNetV1
from lib.nets.fasterrcnn import FasterRCNN


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--batch',
                        dest='batch_size',
                        help='optional batch size',
                        default=1,
                        type=int)
    parser.add_argument('--log',
                        dest='log_path',
                        help='logging for save train',
                        default="./logs/",
                        type=str)
    parser.add_argument('--net',
                        dest='net',
                        help='vgg16, res18, res34,res50, res101, res152, mobilenetV1',
                        default='vgg16',
                        type=str)
    parser.add_argument('--set',
                        dest='set_cfgs',
                        help='set config keys',
                        default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    # print(filename)
    filename = os.path.join(
        "E:/Learn/faster-rcnn/faster-rcnn.pytorch-pytorch-1.0/data/VOCdevkit2007/VOC2007/Annotations",
        filename)
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        objects.append(obj_struct)

    return objects


class voc_eval(object):
    def __init__(self, image_ids, all_boxes, output_dir):
        super().__init__()
        self._comp_id = 'comp4'
        self._salt = str(uuid.uuid4())
        self.output_dir = output_dir
        self.image_ids = image_ids
        self.all_boxes = all_boxes

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            return ap

    def voc_eval(self,
                 detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])

        Top level function that does the PASCAL VOC evaluation.

        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if not os.path.isfile(cachefile):
            # load annotations
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = parse_rec(annopath.format(imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                try:
                    recs = pickle.load(f)
                except:
                    recs = pickle.load(f, encoding='bytes')

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if BB.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)

                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if cfg.use_salt else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_test' + '_{:s}.txt'
        output_path = os.path.join(self.output_dir, "Main")
        if not tf.io.gfile.exists(output_path):
            tf.io.gfile.makedirs(output_path)
        path = os.path.join(output_path, filename)
        return path

    def _write_voc_results_file(self):
        for cls_ind, cls in enumerate(cfg.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_ids):
                    dets = self.all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1,
                            dets[k, 3] + 1))

    def _do_python_eval(self):
        annopath = os.path.join(cfg.data_dir, 'Annotations', '{0:s}.xml')
        imagesetfile = os.path.join(cfg.data_dir, 'ImageSets', 'Main', "val" + '.txt')
        cachedir = os.path.join(cfg.data_dir, 'annotations_cache')

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(2007) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(cfg.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = self.voc_eval(filename,
                                          annopath,
                                          imagesetfile,
                                          cls,
                                          cachedir,
                                          ovthresh=0.5,
                                          use_07_metric=use_07_metric)
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self):
        self._write_voc_results_file()
        self._do_python_eval()
        if cfg.clean_up:
            for cls in cfg.classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)


def test_net(args, dataset, model):
    """Test a Fast R-CNN network on an image database."""
    num_images = dataset[1]
    num_classes = len(cfg.classes)
    image_ids = []
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(cfg.classes))]

    output_dir = os.path.join(cfg.output_dir, "results")
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    test_iterator = iter(dataset[0])
    for i in range(num_images):
        image_id, image, bboxes, labels, height, width = test_iterator.get_next()
        image, bboxes, img_scale = process_data(image, height, width, bboxes)
        image_ids.append(image_id)
        _t['im_detect'].tic()
        boxes, scores = model.predict(image, img_scale)
        _t['im_detect'].toc()

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > cfg.test_thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]

            # keep = nms(cls_dets, cfg.TEST.NMS)
            keep = tf.image.non_max_suppression(cls_boxes,
                                                cls_scores,
                                                max_output_size=-1,
                                                iou_threshold=cfg.nms_thresh)
            cls_dets = np.hstack((cls_boxes[keep], cls_scores[keep, np.newaxis])).astype(np.float32,
                                                                                         copy=False)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if cfg.test_max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > cfg.test_max_per_image:
                image_thresh = np.sort(image_scores)[-cfg.test_max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images,
                                                            _t['im_detect'].average_time,
                                                            _t['misc'].average_time))
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    test_eval = voc_eval(image_ids, all_boxes, output_dir)
    test_eval.evaluate_detections()


if __name__ == "__main__":
    devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(devices[0], True)
    args = parse_args()

    print('Called with args:')
    print(args)

    # 输出configuration
    config = cfg()
    config.display()

    # load network
    if args.net == 'vgg16':
        net = VGG16()
    elif args.net == 'res18':
        net = ResNetV1(num_layers=18)
    elif args.net == 'res34':
        net = ResNetV1(num_layers=34)
    elif args.net == 'res50':
        net = ResNetV1(num_layers=50)
    elif args.net == 'res101':
        net = ResNetV1(num_layers=101)
    elif args.net == 'res152':
        net = ResNetV1(num_layers=152)
    elif args.net == 'mobilenetV1':
        net = MobileNetV1()
    else:
        raise NotImplementedError

    net = VGG16()
    model = FasterRCNN(net)
    model.load_weights('./logs/20210612-114230/frcnn.ckpt')

    val_dir = os.path.join(cfg.output_dir, "%s" % cfg.split_map[1])
    # test_dir = os.path.join(cfg.output_dir, "%s" % cfg.split_map[2])

    val_file_names = [os.path.join(val_dir, i) for i in os.listdir(val_dir)]

    validation_dataset = create_dataset(val_file_names,
                                        batch_size=args.batch_size,
                                        is_shuffle=False,
                                        n_repeats=-1)  # val
    test_net(args, validation_dataset, model)
