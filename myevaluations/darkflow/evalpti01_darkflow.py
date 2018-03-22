import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse, glob, os, sys, json
import numpy as np
from chainercv.datasets.voc import voc_utils
from PIL import Image
import random
from tqdm import tqdm
from myevaluations.metrics.metrics import PTI01Metrics

class DarkflowEvalPTI01():
    def __init__(self, groundtruthpath, predictpath, imagespath, metric='all', limit=None):

        self.groundtruthpath = groundtruthpath
        self.predictpath = predictpath
        self.imagespath = imagespath
        self.metric = metric
        self.limit = limit

        if(self.groundtruthpath == '' or self.predictpath == ''):
            raise Exception('missing pti database paths')

    def buildRect(self, b, color):
        topleft_y = b[0]
        topleft_x = b[1]
        bottomright_y = b[2]
        bottomright_x = b[3]
        return patches.Rectangle((topleft_x,topleft_y),bottomright_x - topleft_x, bottomright_y - topleft_y,linewidth=1,edgecolor=color,facecolor='none')

    def eval(self):

        #default yolo labeling format.
        groundtruths = glob.glob(os.path.join(self.groundtruthpath, '**/*.txt'), recursive=True)
        #darkflow prediction output.
        predictions = glob.glob(os.path.join(self.predictpath, '**/*.json'), recursive=True)
        if self.limit:
            predictions = predictions[0:self.limit]

        gt_size = len(groundtruths)
        pred_size = len(predictions)
        print("{} prediction files | {} groundtruth files".format(pred_size,gt_size))

        pred_bboxes = []
        pred_labels = []
        pred_scores = []

        gt_bboxes = []
        gt_labels = []
        images = []

        for pred_file_path in predictions:
            # print(pred_file_path)
            #remove heading
            base_pred_file_path = pred_file_path.replace(self.predictpath,'')
            # print(base_pred_file_path)

            gt_file_path = os.path.join(self.groundtruthpath, base_pred_file_path).replace('.json','.txt')

            if not os.path.exists(gt_file_path):
                print('GT file does not exist. Tried to open [{}]'.format(gt_file_path))
                sys.exit()

            with open(gt_file_path) as gt_file:
                bboxes = []
                labels = []
                for index,line in enumerate(gt_file):
                    if index == 0:
                        continue
                    gt = list(map(int,line.split())) #[min_x, min_y, max_x, max_y]
                    #must be ('ymin', 'xmin', 'ymax', 'xmax')
                    bboxes.append([gt[1],gt[0],gt[3],gt[2]])
                    labels.append(voc_utils.voc_bbox_label_names.index('person'))
                # print('bboxes',bboxes)
                if len(bboxes) > 0:
                    bboxes = np.stack(bboxes).astype(np.float32)
                    labels = np.stack(labels).astype(np.int32)
                else:
                    bboxes = np.ndarray(shape=(0), dtype=np.float32)
                    labels = np.ndarray(shape=(0), dtype=np.int32)

                if self.imagespath:
                    images.append(os.path.join(self.imagespath, base_pred_file_path).replace('.json','.jpg'))

                gt_bboxes.append(bboxes)
                gt_labels.append(labels)

            with open(pred_file_path) as pred_file:
                j_data = json.load(pred_file)
                # print(j_data)
                bboxes = []
                labels = []
                scores = []
                for predicted_obj in j_data:
                    #must be ('ymin', 'xmin', 'ymax', 'xmax')
                    if predicted_obj['label'] != 'person':
                        continue
                    bboxes.append([
                        predicted_obj['topleft']['y'],
                        predicted_obj['topleft']['x'],
                        predicted_obj['bottomright']['y'],
                        predicted_obj['bottomright']['x']
                        ])
                    labels.append(voc_utils.voc_bbox_label_names.index('person'))
                    scores.append(predicted_obj['confidence'])
                if len(bboxes) > 0:
                    bboxes = np.stack(bboxes).astype(np.float32)
                    labels = np.stack(labels).astype(np.int32)
                    scores = np.stack(scores).astype(np.float32)
                else:
                    bboxes = np.ndarray(shape=(0), dtype=np.float32)
                    labels = np.ndarray(shape=(0), dtype=np.int32)
                    scores = np.ndarray(shape=(0), dtype=np.float32)

                pred_bboxes.append(bboxes)
                pred_labels.append(labels)
                pred_scores.append(scores)

        # pred_bboxes = np.stack(pred_bboxes).astype(np.float32)
        # pred_labels = np.stack(pred_labels).astype(np.int32)
        # pred_scores = np.stack(pred_scores).astype(np.float32)

        pred_bboxes = np.asarray(pred_bboxes)
        pred_labels = np.asarray(pred_labels)
        pred_scores = np.asarray(pred_scores)

        gt_bboxes = np.asarray(gt_bboxes)
        gt_labels = np.asarray(gt_labels)

        metrics = PTI01Metrics((pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels), metric=self.metric, database_name='PTI01', limit=self.limit)
        metrics.calc()


        if self.imagespath:
            print('Gerenrating plots...')
            for x in tqdm(range(100), total=100):
                target_image = random.randint(0, len(gt_bboxes)-1)
                curr_img = images[target_image]
                im = np.array(Image.open(curr_img), dtype=np.uint8)
                fig,ax = plt.subplots(1)
                ax.imshow(im)
                for b in gt_bboxes[target_image]:
                    rect = self.buildRect(b,'g')
                    ax.add_patch(rect)
                for b in pred_bboxes[target_image]:
                    rect = self.buildRect(b,'r')
                    ax.add_patch(rect)
                plt.axis('off')
                fig_dir = 'plottings'
                os.makedirs(fig_dir, exist_ok=True)
                fig_path = os.path.join(fig_dir, curr_img.replace(self.imagespath,'').replace('/',''))
                plt.savefig(fig_path, bbox_inches='tight')
                plt.show()
                plt.close()
