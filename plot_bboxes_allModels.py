import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime, os
import numpy as np
from tqdm import tqdm
from PIL import Image

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

fig_dir = 'plottings/all_models/{}_{}_{}/'.format('PTI01', len(fr_images),now)
os.makedirs(fig_dir, exist_ok=True)
print('Saving plots in {}.'.format(fig_dir))

for im_index in tqdm(range(len(fr_images)), total=len(fr_images)):
    curr_img = fr_images[im_index].replace('\n','').strip()
    curr_img = os.path.join(IMAGES_PATH,curr_img)
    im = np.array(Image.open(curr_img), dtype=np.uint8)
    fig,(fr_ax, s3_ax, s5_ax, yl_ax) = plt.subplots(nrows=1,ncols=4)

    fr_ax.imshow(im)
    for b in fr_gt_bboxes[im_index]:
        rect = buildRect(b,'g')
        fr_ax.add_patch(rect)
    for b in fr_pred_bboxes[im_index]:
        rect = buildRect(b,'r')
        fr_ax.add_patch(rect)
    fr_ax.axis('off')

    s3_ax.imshow(im)
    for b in s3_gt_bboxes[im_index]:
        rect = buildRect(b,'g')
        s3_ax.add_patch(rect)
    for b in s3_pred_bboxes[im_index]:
        rect = buildRect(b,'r')
        s3_ax.add_patch(rect)
    s3_ax.axis('off')

    s5_ax.imshow(im)
    for b in s5_gt_bboxes[im_index]:
        rect = buildRect(b,'g')
        s5_ax.add_patch(rect)
    for b in s5_pred_bboxes[im_index]:
        rect = buildRect(b,'r')
        s5_ax.add_patch(rect)
    s5_ax.axis('off')

    yl_ax.imshow(im)
    for b in yl_gt_bboxes[im_index]:
        rect = buildRect(b,'g')
        yl_ax.add_patch(rect)
    for b in yl_pred_bboxes[im_index]:
        rect = buildRect(b,'r')
        yl_ax.add_patch(rect)
    yl_ax.axis('off')
    # yl_ax.subplots_adjust(wspace=0, hspace=0)

    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    # plt.figtext(1, 1, 'img:{}'.format(curr_img))

    fig_path = os.path.join(fig_dir, curr_img.replace(IMAGES_PATH,'').replace('/',''))
    fig.set_size_inches(20,20)
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    # print(fr_images[50],s3_images[50],s5_images[50],yl_images[50])
    # break
