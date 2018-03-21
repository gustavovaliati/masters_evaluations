import numpy as np
import os
import warnings
# import xml.etree.ElementTree as ET
import glob

import chainer

from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class PTI01BboxDataset(chainer.dataset.DatasetMixin):
    def __init__(self, imagespath, labelspath, limit=None):
        self.limit = None if not limit else int(limit)
        self.imagespath = imagespath
        self.labelspath = labelspath
        if(self.imagespath == '' or self.labelspath == ''):
            raise Exception('missing pti database paths')

        self.file_names = glob.glob(os.path.join(self.imagespath, '**/*.jpg'), recursive=True)
        self.file_names.sort()

    def __len__(self):
        if self.limit == None:
            return len(self.file_names)
        else:
            return self.limit

    def get_example(self, i):
        image_ = self.file_names[i]
        bbox = []
        label = []

        ground_truth_file_path = image_.replace('.jpg', '.txt').replace(self.imagespath,self.labelspath)
        with open(ground_truth_file_path,'r') as ground_truth_file:
            for index,line in enumerate(ground_truth_file):
                if index == 0:
                    continue
                gt = list(map(int,line.split())) #[min_x, min_y, max_x, max_y]
                #must be ('ymin', 'xmin', 'ymax', 'xmax')
                bbox.append([gt[1],gt[0],gt[3],gt[2]])
                label.append(voc_utils.voc_bbox_label_names.index('person'))
        # print('bbox',bbox)
        if len(bbox) > 0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        else:
            bbox = np.ndarray(shape=(0), dtype=np.float32)
            label = np.ndarray(shape=(0), dtype=np.int32)
        img = read_image(image_, color=True)

        return img, bbox, label
