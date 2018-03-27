'''
Temporary script for generating the txt files with gt and predicitons in mot format from chainercv data.
'''
from myevaluations.backup import Backup
import pandas as pd
import os, datetime

class Parser:
    def __init__(self):
        self.backuper = Backup()

    def convert_chainerdata_to_motdata_files(self, loadfrom=None, dataset_name=None, model_name=None):
        """
        Converts the gt and ts data format from chainer style to the motchallenge input file format.
        The generated files are statically defined in the motchallenge-devkit-aux dir.
        """
        if not loadfrom:
            raise Exception('Missing loadfrom.')
        if not dataset_name:
            raise Exception('Missing dataset_name.')
        if not model_name:
            raise Exception('Missing model_name.')

        splitted_loadfrom = loadfrom.split('_')
        loadfrom_model_name = splitted_loadfrom[1].lower()
        loadfrom_dataset_name = splitted_loadfrom[2].lower()


        '''
        Checks if the backup file is of the same dataset and model statically defined in the eval.py.
        '''
        if not dataset_name.lower() == loadfrom_dataset_name:
            raise Exception('dataset_name missmatch. The eval.py statically defines the dataset_name and should match to the backup files. Given {} | Expected: {}'.format(dataset_name.lower(), loadfrom_dataset_name))

        if not model_name.lower() == loadfrom_model_name:
            raise Exception('model_name missmatch. The eval.py statically defines the model_name and should match to the backup file. Given {} | Expected: {}'.format(model_name.lower(), loadfrom_model_name))


        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = self.backuper.load(loadfrom)

        now = datetime.datetime.now()
        gt_dir_path = './motchallenge-devkit-aux/datasets/train/{}/gt/'.format(dataset_name)
        os.makedirs(gt_dir_path, exist_ok=True)
        gt_path = os.path.join(gt_dir_path, 'gt_{}.txt'.format(now.strftime('%Y%m%d_%H%M%S')))
        print("GT file path: {}".format(gt_path))

        pred_dir_path = './motchallenge-devkit-aux/res/{}/{}/data'.format(dataset_name,model_name)
        os.makedirs(pred_dir_path, exist_ok=True)
        pred_path = os.path.join(pred_dir_path, '{}_{}.txt'.format(dataset_name, now.strftime('%Y%m%d_%H%M%S')))
        print("PRED file path: {}".format(pred_path))

        with open(gt_path, 'w') as gt_f:
            for frame_id, bboxes in enumerate(gt_bboxes):
                for bbox_id, bbox in enumerate(bboxes):
                    #bbox = ('ymin', 'xmin', 'ymax', 'xmax')
                    row = []

                    #FrameId
                    row.append(str(frame_id + 1)) #increase 1 for matlab style
                    #Id
                    # row.append('-1')
                    row.append(str(frame_id + 1))
                    #X | bb_left
                    row.append(str(bbox[1]))
                    #Y | bb_top
                    row.append(str(bbox[0]))
                    #Width
                    row.append(str(bbox[3]-bbox[1]+1))
                    #Height
                    row.append(str(bbox[2]-bbox[0]+1))
                    #acoording to the file examples, there are no confidence column in the GT file. But there are X,Y,Z in the end with -1
                    row.append('1') # I dont know what is this. The original gt file has zeros or ones. Here both gives the same result.
                    row.append('1') #classId = 1 for person
                    row.append('1') #minvis must be greater than 0.5

                    gt_f.write(",".join(row) + "\n")

        with open(pred_path, 'w') as pred_f:
            for frame_id, bboxes in enumerate(pred_bboxes):
                for bbox_id, bbox in enumerate(bboxes):
                    #bbox = ('ymin', 'xmin', 'ymax', 'xmax')
                    row = []

                    #FrameId
                    row.append(str(frame_id + 1)) #increase 1 for matlab style
                    #Id
                    row.append('-1')
                    #X | bb_left
                    row.append(str(bbox[1]))
                    #Y | bb_top
                    row.append(str(bbox[0]))
                    #Width
                    row.append(str(bbox[3]-bbox[1]+1))
                    #Height
                    row.append(str(bbox[2]-bbox[0]+1))
                    #Confidence
                    row.append(str(pred_scores[frame_id][bbox_id]))

                    pred_f.write(",".join(row) + "\n")


    def convert_chainerdata_to_motdata_pandas(self, chainerdata):
        """
        Converts the gt and ts data format from chainer style to the motchallenge style in pandas dataframe.
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
