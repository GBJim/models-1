# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import ds_utils as ds_utils
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import json
from vatic import create_tf_example
# COCO API
from pycocotools.coco import COCO
from scipy.misc import imread
import tensorflow as tf


def _filter_crowd_proposals(roidb, crowd_thresh):
    """
    Finds proposals that are inside crowd regions and marks them with
    overlap = -1 (for all gt rois), which means they will be excluded from
    training.
    """
    for ix, entry in enumerate(roidb):
        overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(overlaps.max(axis=1) == -1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        iscrowd = [int(True) for _ in xrange(len(crowd_inds))]
        crowd_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        overlaps[non_gt_inds[bad_inds], :] = -1
        roidb[ix]['gt_overlaps'] = scipy.sparse.csr_matrix(overlaps)
    return roidb

class coco():
    def __init__(self, image_set, year, CLS_mapper={}, data_path = "/root/data/coco"):
        
        # COCO specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'min_size' : 2}
        # name, paths
        self.is_video = False
        self.CLS_mapper = CLS_mapper
        self._year = year
        self._image_set = image_set
        self._data_path = data_path
        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self._COCO.getCatIds()))
        
        
   
        self._image_index = self._load_image_set_index()
        self.size = len(self._image_index)
        # Default to roidb handler
        #self.set_proposal_method('selective_search')
        self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014' : 'val2014',          # 5k val2014 subset
            'valminusminival2014' : 'val2014',  # val2014 \setminus minival2014
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if self._view_map.has_key(coco_name)
                           else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
                             else 'image_info'
        return osp.join(self._data_path, 'annotations',
                        prefix + '_' + self._image_set + self._year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids
    
    def dumpTFRecord(self, output_path, category_name2id):
        writer = tf.python_io.TFRecordWriter(output_path)
        img_info = {}

        if self.is_video:
            img = imread(self.image_path_at(0))
            
            img_info["height"] = img.shape[0] 
            img_info['width'] = img.shape[1]
            img_info["format"] = self.image_path_at(0).split('.')[-1]
        
        for i in range(self.size):
            img_info['path'] = self.image_path_at(i)
            if not self.is_video:
                img = imread(self.image_path_at(i))
                img_info["height"] = img.shape[0] 
                img_info['width'] = img.shape[1]
                img_info["format"] = self.image_path_at(i).split('.')[-1]
                
            bboxes = self.bbox_at(i)
            tf_record = create_tf_example(bboxes, img_info, category_name2id, self.CLS_mapper)
            writer.write(tf_record.SerializeToString())
        writer.close()
        print("Tensorflow Record has been writen into {}".format(output_path))
     
    def bbox_at(self, i):
        index = self._image_index[i]
        coco_annotation = self._load_coco_annotation(index)
        bboxes = []
        for i, cls in enumerate(coco_annotation["gt_classes"]):           
            if (coco_annotation["gt_overlaps"])[i][cls] > 0:
                x1, y1, x2, y2 = coco_annotation["boxes"][i]
                label = self._classes[cls]
                width = x2 - x1
                height = y2 - y1
                bbox = {"x1":x1, "y1":y1, "width":width, "height":height, "label":label}
                bboxes.append(bbox)
        return bboxes                                
             
                                    
            


    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('COCO_' + self._data_name + '_' +
                     str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path


    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        #overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_box_file(self, index):
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        file_name = ('COCO_' + self._data_name +
                     '_' + str(index).zfill(12) + '.mat')
        return osp.join(file_name[:14], file_name[:22], file_name)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print ('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh)
        print '{:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print '{:.1f}'.format(100 * ap)

        print '~~~~ Summary metrics ~~~~'
        coco_eval.summarize()

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in xrange(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                          self.num_classes - 1)
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print 'Writing results json to {}'.format(res_file)
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
