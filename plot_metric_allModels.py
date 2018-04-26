import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime, os
import numpy as np
from tqdm import tqdm
from PIL import Image

from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec
from chainercv.datasets.voc import voc_utils

FASTER_RCNN_BACKUP='backups/pred_fasterrcnn_pti01_7927_20180409_225704.npy'
SSD300_BACKUP='backups/pred_ssd300_pti01_7927_20180409_231419.npy'
SSD512_BACKUP='backups/pred_ssd512_pti01_7927_20180409_224650.npy'
YOLOV2_BACKUP='backups/pred_yolov2_pti01_7927_20180407_112556.npy'

IMAGES_PATH = '/home/gustavo/workspace/bbox-grv/Images/001/'

def buildRect(b, color):
    topleft_y = b[0]
    topleft_x = b[1]
    bottomright_y = b[2]
    bottomright_x = b[3]
    return patches.Rectangle((topleft_x,topleft_y),bottomright_x - topleft_x, bottomright_y - topleft_y,linewidth=1,edgecolor=color,facecolor='none')

print('Reading predictions backup...')
fr_pred_bboxes,fr_pred_labels,fr_pred_scores,fr_gt_bboxes,fr_gt_labels,fr_images = np.load(FASTER_RCNN_BACKUP)
s3_pred_bboxes,s3_pred_labels,s3_pred_scores,s3_gt_bboxes,s3_gt_labels,s3_images = np.load(SSD300_BACKUP)
s5_pred_bboxes,s5_pred_labels,s5_pred_scores,s5_gt_bboxes,s5_gt_labels,s5_images = np.load(SSD512_BACKUP)
tmp_yl_pred_bboxes,tmp_yl_pred_labels,tmp_yl_pred_scores,tmp_yl_gt_bboxes,tmp_yl_gt_labels,tmp_yl_images = np.load(YOLOV2_BACKUP)
print('Sucessfuly loaded')
print('Fixing yolo positions')

yl_pred_bboxes,yl_pred_labels,yl_pred_scores,yl_gt_bboxes,yl_gt_labels,yl_images = [],[],[],[],[],[]
for correct_index, fr_img in enumerate(fr_images):

    curr_index = tmp_yl_images.tolist().index(fr_img.replace('\n',''))

    yl_pred_bboxes.append(tmp_yl_pred_bboxes[curr_index])
    yl_pred_labels.append(tmp_yl_pred_labels[curr_index])
    yl_pred_scores.append(tmp_yl_pred_scores[curr_index])
    yl_gt_bboxes.append(tmp_yl_gt_bboxes[curr_index])
    yl_gt_labels.append(tmp_yl_gt_labels[curr_index])
    yl_images.append(tmp_yl_images[curr_index])


print('Gerenrating plots...')
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

fig_dir = 'plottings/metrics/all_models/{}_{}_{}/'.format('PTI01', len(fr_images),now)
os.makedirs(fig_dir, exist_ok=True)

print('Saving metric in {}.'.format(fig_dir))

print('YOLOv2')
yl_prec, yl_rec = calc_detection_voc_prec_rec(
yl_pred_bboxes, yl_pred_labels, yl_pred_scores,
yl_gt_bboxes, yl_gt_labels, gt_difficults=None, iou_thresh=0.5)
yl_person_prec = yl_prec[voc_utils.voc_bbox_label_names.index('person')]
yl_person_rec = yl_rec[voc_utils.voc_bbox_label_names.index('person')]
print("Avg prec {}, Avg rec {}".format(np.average(yl_person_prec), np.average(yl_person_rec)))
plt.step(yl_person_rec, yl_person_prec, label='YOLOv2')


print('SSD512')
s5_prec, s5_rec = calc_detection_voc_prec_rec(
    s5_pred_bboxes, s5_pred_labels, s5_pred_scores,
    s5_gt_bboxes, s5_gt_labels, gt_difficults=None, iou_thresh=0.5)
s5_person_prec = s5_prec[voc_utils.voc_bbox_label_names.index('person')]
s5_person_rec = s5_rec[voc_utils.voc_bbox_label_names.index('person')]
print("Avg prec {}, Avg rec {}".format(np.average(s5_person_prec), np.average(s5_person_rec)))
plt.step(s5_person_rec, s5_person_prec, label='SSD512')

print('Faster R-CNN')
fr_prec, fr_rec = calc_detection_voc_prec_rec(
fr_pred_bboxes, fr_pred_labels, fr_pred_scores,
fr_gt_bboxes, fr_gt_labels, gt_difficults=None, iou_thresh=0.5)
fr_person_prec = fr_prec[voc_utils.voc_bbox_label_names.index('person')]
fr_person_rec = fr_rec[voc_utils.voc_bbox_label_names.index('person')]
print("Avg prec {}, Avg rec {}".format(np.average(fr_person_prec), np.average(fr_person_rec)))
plt.step(fr_person_rec, fr_person_prec, label='Faster R-CNN')

print('SSD300')
s3_prec, s3_rec = calc_detection_voc_prec_rec(
s3_pred_bboxes, s3_pred_labels, s3_pred_scores,
s3_gt_bboxes, s3_gt_labels, gt_difficults=None, iou_thresh=0.5)
s3_person_prec = s3_prec[voc_utils.voc_bbox_label_names.index('person')]
s3_person_rec = s3_rec[voc_utils.voc_bbox_label_names.index('person')]
print("Avg prec {}, Avg rec {}".format(np.average(s3_person_prec), np.average(s3_person_rec)))
plt.step(s3_person_rec, s3_person_prec, label='SSD300')

plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10))
plt.savefig(os.path.join(fig_dir, 'recprec__precision_recall_curve.jpg'), bbox_inches='tight')
plt.show()
plt.close()
