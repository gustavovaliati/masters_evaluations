import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime, random, os
import numpy as np
from tqdm import tqdm
from PIL import Image

class Plottings:
    def __init__(self,implot,model,dataset_name,imagespath):
        self.implot = implot
        self.imagespath = imagespath
        self.model = model
        self.dataset_name = dataset_name

    def buildRect(self, b, color):
        topleft_y = b[0]
        topleft_x = b[1]
        bottomright_y = b[2]
        bottomright_x = b[3]
        return patches.Rectangle((topleft_x,topleft_y),bottomright_x - topleft_x, bottomright_y - topleft_y,linewidth=1,edgecolor=color,facecolor='none')

    def plot(self, gt_bboxes, pred_bboxes, images,person_prec=[],person_rec=[]):
        plot_text = False
        if len(person_prec) > 0 and len(person_rec) > 0:
            plot_text = True

        print('Gerenrating plots...')
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        rand = self.implot['rand']
        count = self.implot['count'] if self.implot['count'] < len(gt_bboxes) else len(gt_bboxes)

        index_pool = range(count)
        if rand:
            print('Plotting bboxes for random images.')
            index_pool = random.sample(range(len(gt_bboxes)),count) #randomly gets non-repeated indexes.

        fig_dir = 'plottings/{}_{}_{}_{}/'.format(self.model.replace('_',''), self.dataset_name, len(gt_bboxes),now)
        os.makedirs(fig_dir, exist_ok=True)
        print('Saving plots in {}.'.format(fig_dir))

        for im_index in tqdm(index_pool, total=count):
            curr_img = images[im_index].replace('\n','').strip()
            curr_img = os.path.join(self.imagespath,curr_img)
            im = np.array(Image.open(curr_img), dtype=np.uint8)
            fig,ax = plt.subplots(1)
            ax.imshow(im)
            for b in gt_bboxes[im_index]:
                rect = self.buildRect(b,'g')
                ax.add_patch(rect)
            for b in pred_bboxes[im_index]:
                rect = self.buildRect(b,'r')
                ax.add_patch(rect)
            plt.axis('off')

            # if plot_text:
                # plt.figtext(0.0, 0.05, 'p:{} r:{} img:{}'.format(person_prec[im_index],person_rec[im_index],curr_img))

            fig_path = os.path.join(fig_dir, curr_img.replace(self.imagespath,'').replace('/',''))
            fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()
