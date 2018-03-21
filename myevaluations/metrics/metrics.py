import numpy as np
from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec
from chainercv.datasets.voc import voc_utils

class PTI01Metrics:
    def __init__(self, data, metric):
        self.metric = metric
        self.data = data

    def calc(self):
        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = self.data

        if self.metric == 'all' or self.metric == 'voc_detection':
            print('Calculating voc_detection ...')
            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                use_07_metric=True)
            print('mAP: {:f}'.format(result['map']))
            print('person mAP: {:f}'.format(result['ap'][voc_utils.voc_bbox_label_names.index('person')]))

        if self.metric == 'all' or self.metric == 'pr_voc_detection':
            print('Calculating pr_voc_detection ...')
            prec, rec = calc_detection_voc_prec_rec(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults=None, iou_thresh=0.5)
            print('Avg person precision: {:f}'.format(np.average(prec[voc_utils.voc_bbox_label_names.index('person')])))
            print('Avg person recall: {:f}'.format(np.average(rec[voc_utils.voc_bbox_label_names.index('person')])))
