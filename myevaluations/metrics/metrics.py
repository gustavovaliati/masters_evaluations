import matplotlib
matplotlib.use('Agg')

import numpy as np
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
import datetime, os

from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec
from chainercv.datasets.voc import voc_utils
from myevaluations.parser import Parser

metric_plot_choices = [
    'all',
    'precision_recall_curve'
    ]

class PTI01Metrics:
    def __init__(self, data, metric, database_name, plottings, model,limit=None):
        self.metric = metric
        self.data = data
        self.database_name = database_name
        self.limit = int(limit) if limit else None
        self.plottings = plottings
        self.model = model
        self.parser = Parser()

    def compare_dataframes(self, gts, ts):
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                print('Comparing {}...'.format(k))
                accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                names.append(k)
            else:
                print('No ground truth for {}, skipping.'.format(k))

        return accs, names

    def convert_chainerdata_to_motdata(self, chainerdata):
        """
        Converts the gt and ts data format from chainer style to the motchallenge style.
        """

        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = chainerdata

        """
        df : pandas.DataFrame
            The returned dataframe has the following columns
                'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
            The dataframe is indexed by ('FrameId', 'Id')
        """

        gt_dict = {'FrameId': [], 'Id': [], 'X': [], 'Y': [], 'Width': [], 'Height': [], 'Confidence': [], 'ClassId': [], 'Visibility': []}
        for frame_id, bboxes in enumerate(gt_bboxes):
            for bbox_id, bbox in enumerate(bboxes):
                #bbox = ('ymin', 'xmin', 'ymax', 'xmax')
                # print('gt', frame_id, bbox_id, bbox)
                gt_dict['FrameId'].append(frame_id)
                gt_dict['Id'].append(frame_id)
                gt_dict['X'].append(bbox[1])
                gt_dict['Y'].append(bbox[0])
                gt_dict['Width'].append(bbox[3]-bbox[1]+1)
                gt_dict['Height'].append(bbox[2]-bbox[0]+1)
                gt_dict['Confidence'].append(1)
                gt_dict['ClassId'].append(-1)
                gt_dict['Visibility'].append(-1)

        gt_df = pd.DataFrame(data=gt_dict)
        gt_df = gt_df.set_index(['FrameId','Id'])

        gt_bboxes_size = len(gt_bboxes)
        pred_dict = {'FrameId': [], 'Id': [], 'X': [], 'Y': [], 'Width': [], 'Height': [], 'Confidence': [], 'ClassId': [], 'Visibility': []}
        for frame_id, bboxes in enumerate(pred_bboxes):
            for bbox_id, bbox in enumerate(bboxes):
                #bbox = ('ymin', 'xmin', 'ymax', 'xmax')
                # print('pred', frame_id, bbox_id, bbox)
                pred_dict['FrameId'].append(frame_id)
                pred_dict['Id'].append(frame_id+gt_bboxes_size)
                pred_dict['X'].append(bbox[1])
                pred_dict['Y'].append(bbox[0])
                pred_dict['Width'].append(bbox[3]-bbox[1]+1)
                pred_dict['Height'].append(bbox[2]-bbox[0]+1)
                pred_dict['Confidence'].append(pred_scores[frame_id][bbox_id])
                pred_dict['ClassId'].append(-1)
                pred_dict['Visibility'].append(-1)

        pred_df = pd.DataFrame(data=pred_dict)
        pred_df = pred_df.set_index(['FrameId','Id'])

        return pred_df, gt_df

    def motmetrics(self, solver=None):

        if solver:
            mm.lap.default_solver = solver

        pred_df, gt_df = self.parser.convert_chainerdata_to_motdata_pandas(self.data)

        limit = self.limit if self.limit else None
        if limit:
            print("motmetrics limit set to {} prediction bboxes. WARNING: USE THIS FOR DEBUG ONLY. THE STATISTICS WILL BECOME INCOHERENT.".format(limit))

        ts = OrderedDict([(self.database_name,pred_df[:limit])])
        gt = OrderedDict([(self.database_name,gt_df)])
        print("Running motmetrics over {} pred bboxes.".format(len(pred_df)))
        print("Running motmetrics over {} gt bboxes.".format(len(gt_df)))

        mh = mm.metrics.create()
        accs, names = self.compare_dataframes(gt, ts)

        print('Running metrics - test 1')

        summary = mh.compute_many(accs, metrics=['num_frames', 'num_detections', 'num_false_positives',
            'num_misses', 'mota', 'motp', 'precision','recall'])
        print(summary)

        # print('Running metrics - test 2')
        # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
        # print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

        print('Completed')

    def plot(self, recall, precision, metric=None):

        if not metric:
            raise Exception('Metric name must be defined for plotting.')
        if not self.plottings:
            raise Exception('No plottings are specified.')

        if 'all' in self.plottings:
            print('Plotting everything.')
        else:
            for pl in self.plottings:
                if not pl in metric_plot_choices:
                    parser.error('The plotting you requested does not exist: {}.'.format(pl))
                    return

        fig_dir = "plottings/{}_{}_{}".format(self.model, self.database_name,
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(fig_dir, exist_ok=True)

        if 'all' in self.plottings:
            self.plottings = metric_plot_choices
            self.plottings.remove('all')

        if 'all' in self.plottings or 'precision_recall_curve' in self.plottings:
            self.plottings.remove('precision_recall_curve')
            print("Plotting {}.".format('precision_recall_curve'))

            print("Min prec {}, Max prec {}, Min rec {}, Max rec {}".format(min(precision),max(precision),min(recall),max(recall)))
            print("Avg prec {}, Avg rec {}".format(np.average(precision), np.average(recall)))

            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2,
                             color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            #           average_precision))

            plt.savefig(os.path.join(fig_dir, metric+'__precision_recall_curve.jpg'), bbox_inches='tight')
            plt.show()
            plt.close()


        if len(self.plottings) > 0:
            raise Exception("There are unknown requested plottings: {}".format(self.plottings))

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

            person_prec = prec[voc_utils.voc_bbox_label_names.index('person')]
            person_rec = rec[voc_utils.voc_bbox_label_names.index('person')]
            print('Avg person precision: {:f}'.format(np.average(person_prec)))
            print('Avg person recall: {:f}'.format(np.average(person_rec)))

            if self.plottings:
                self.plot(recall=person_rec, precision=person_prec, metric='pr_voc_detection')

        if self.metric == 'all' or self.metric == 'mot':
            print('Calculating mot_metrics ...')
            self.motmetrics()
