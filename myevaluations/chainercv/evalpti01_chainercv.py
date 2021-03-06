import argparse
from itertools import tee

import chainer
from chainer import iterators
from chainercv.datasets import voc_bbox_label_names
# from chainercv.datasets import VOCBboxDataset
from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec
from chainercv.links import FasterRCNNVGG16
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.utils import apply_prediction_to_iterator
from chainercv.utils import ProgressHook

from myevaluations.chainercv.ptidatasets import PTI01BboxDataset
from myevaluations.backup import Backup
from myevaluations.metrics.metrics import PTI01Metrics

class ChainercvEvalPTI01():
    def __init__(self, groundtruthpath, imagespath, plot, pretrained_model=None,
        model='ssd300', limit=None, gpu=-1, batchsize=32, metric='all', loadfrom=None):

        self.model = model
        self.limit = limit
        self.metric = metric
        self.imagespath = imagespath
        self.labelspath = groundtruthpath
        self.pretrained_model = pretrained_model
        self.gpu = gpu
        self.batchsize = batchsize
        self.backuper = Backup()
        self.loadfrom = loadfrom
        self.plot = plot

    def eval(self):

        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = [],[],[],[],[]

        if self.loadfrom:
             pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = self.backuper.load(self.loadfrom)
             print("The lists have been loaded with {} objects.".format(len(pred_bboxes)))
        else:
            if self.model == 'faster_rcnn':
                if self.pretrained_model:
                    model = FasterRCNNVGG16(
                        n_fg_class=20,
                        pretrained_model=self.pretrained_model)
                else:
                    model = FasterRCNNVGG16(pretrained_model='voc07')
            elif self.model == 'ssd300':
                if self.pretrained_model:
                    model = SSD300(
                        n_fg_class=20,
                        pretrained_model=self.pretrained_model)
                else:
                    model = SSD300(pretrained_model='voc0712')
            elif self.model == 'ssd512':
                if self.pretrained_model:
                    model = SSD512(
                        n_fg_class=20,
                        pretrained_model=self.pretrained_model)
                else:
                    model = SSD512(pretrained_model='voc0712')

            if self.gpu >= 0:
                chainer.cuda.get_device_from_id(self.gpu).use()
                model.to_gpu()

            model.use_preset('evaluate')

            # dataset = VOCBboxDataset(
            #     year='2007', split='test', use_difficult=True, return_difficult=True)

            dataset = PTI01BboxDataset(limit=self.limit, imagespath=self.imagespath,
                labelspath=self.labelspath)
            iterator = iterators.SerialIterator(
                dataset, self.batchsize, repeat=False, shuffle=False)

            imgs, pred_values, gt_values = apply_prediction_to_iterator(
                model.predict, iterator, hook=ProgressHook(len(dataset)))
            # delete unused iterator explicitly
            del imgs

            pred_bboxes, pred_labels, pred_scores = pred_values
            gt_bboxes, gt_labels = gt_values

            pred_bboxes = list(pred_bboxes)
            pred_labels = list(pred_labels)
            pred_scores = list(pred_scores)
            gt_bboxes = list(gt_bboxes)
            gt_labels = list(gt_labels)

            self.backuper.save(self.model, 'pti01', len(dataset), (pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels))

        if(len(gt_bboxes) == 0):
            print('Warning: gt_bboxes is empty')
            gt_bboxes, gt_label = [],[]

        # print('{} {} {} {} {}'.format(len(pred_bboxes),len(pred_labels),len(pred_scores),len(gt_bboxes),len(gt_labels)))

        metrics = PTI01Metrics((pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels), model=self.model, metric=self.metric,
            database_name='PTI01', limit=self.limit, plottings=self.plot)
        metrics.calc()
