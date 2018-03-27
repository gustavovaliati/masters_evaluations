import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

new_gt_f_path = '/home/gustavo/workspace/datasets/MOT17DetLabels/train/MOT17-02/gt/gt-modified.txt'

with open('/home/gustavo/workspace/datasets/MOT17DetLabels/train/MOT17-02/gt/gt.txt','r') as original_gt_f:
    with open(new_gt_f_path, 'w') as new_gt_f:
        for index,line in enumerate(original_gt_f):
            line = line.split(',')
            line[1] = str(index)
            new_gt_f.write(','.join(line))


gt_df = mm.io.loadtxt(new_gt_f_path)

# for row in gt_df.itertuples():
#     print(row['Index'])
#     FrameId, Id = row[0]
#     print(gt_df.at[row[0],row[0]])
#     gt_df.at[row[0],'ClassId'] = -1
#     print(gt_df.at[row[0],'ClassId'])

gt = OrderedDict([('MOT17-02', gt_df)])
ts = OrderedDict([('MOT17-02', mm.io.loadtxt('~/workspace/motchallenge-devkit/res/MOT17Det/DPM/data/MOT17-02.txt'))])

# mh = mm.metrics.create()
# accs, names = compare_dataframes(gt, ts)
#
#
# summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
# print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

mh = mm.metrics.create()
accs, names = compare_dataframes(gt, ts)

print('Running metrics - test 1')

summary = mh.compute_many(accs, metrics=['num_frames', 'num_detections', 'num_false_positives',
    'num_misses','precision','recall', 'mota', 'motp'])
print(summary)
