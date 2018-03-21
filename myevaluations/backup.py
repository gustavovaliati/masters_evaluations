import os, datetime
import numpy as np
from itertools import tee

class Backup():
    def save(self, model, dataset_name, dataset_size, data):
        f_path = "./backups/pred_{}_{}_{}_{}".format(model, dataset_name, dataset_size,
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        if os.path.exists(f_path):
            raise Exception('Backup file {} does already exist.'.format(f_path))

        print('Saving predictions...')
        # pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = _data

        '''
        Due this error:
            >> "TypeError: can't pickle generator objects"
        we cannot save the objects directly. It is needed to transform them to
        common objects in the loop below.
        '''
        # looper = [pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels]
        # obj_tosave = []
        # for i, arr in enumerate(looper):
        #     obj_tosave.insert(i,[])
        #     for x in list(arr):
        #         obj_tosave[i].append(x)

        np.save(f_path, data)
        print('Saved to {}.'.format(f_path))
        # with open(f_path, 'w') as f:
        #     pickle.dump(data, f)

    def load(self, f_path):
        if not os.path.exists(f_path):
            raise Exception('Backup file {} does not exist.'.format(f_path))

        print('Reading predictions backup...')
        data = np.load(f_path)
        print('Sucessfuly loaded from {}.'.format(f_path))
        return data
        # with open(f_path, 'r') as f:
            # data = pickle.load(f)
