import argparse
from myevaluations.darkflow.evalpti01_darkflow import DarkflowEvalPTI01
from myevaluations.chainercv.evalpti01_chainercv import ChainercvEvalPTI01
from myevaluations.metrics.metrics import metric_plot_choices
from myevaluations.parser import Parser

#python3 eval.py --model yolov2 --loadfrom ~/workspace/darkflow/jsonoutput/  --gtpath ~/workspace/datasets/pti/PTI01-bbox-labels/
#python3 eval.py --model ssd512 --gpu 0 --impath ~/workspace/datasets/pti/PTI01/ --gtpath ~/workspace/datasets/pti/PTI01-bbox-labels/
#python3 eval.py --model yolov2 --parse_to_mot_file --loadfrom backups/pred_yolov2_pti01_7927_20180326_223828.npy

def main():
    parser = argparse.ArgumentParser(description="An evaluator for the masters degree experiments.")
    parser.add_argument(
        '--model', choices=('faster_rcnn', 'ssd300', 'ssd512', 'yolov2'),
        default=False, required = True)
    parser.add_argument(
        '--metric', choices=('voc_detection','mot','all'),
        default='all')

    parser.add_argument('--plot', nargs='*',
        help='Choose the plottings for the metrics. They will be saved in the plottings/ folder.'
        ' You can choose many at once. Use like --plot this_plot that_plot .'
        ' Available choices are: {}.'.format(metric_plot_choices))

    parser.add_argument('--loadfrom',
        help=   "Load previously saved predictions instead of evaluating images again."
                " For yolov2, select the darkflow's jsonoutput folder."
                " For faster_rcnn,ssd300,ssd512 choose the 'pred*.npy' file into the backup folder.")
    parser.add_argument('--limit', default=None, type=int,
        help=   "Limits the number of images from the dataset."
                "Does not work for limiting metrics yet.")

    # chainercv arguments
    parser.add_argument('--pretrained_model', help="chainercv argument")
    parser.add_argument('--gpu', type=int, default=-1, help="chainercv argument")
    parser.add_argument('--batchsize', type=int, default=32, help="chainercv argument")
    # parser.add_argument('--ptiimagespath')
    # parser.add_argument('--ptilabelspath')

    #darkflow arguments
    #darkflow must be used from the repository: github.com/gustavovaliati/darkflow
    #command: flow --imgdir ~/workspace/datasets/pti/PTI01/ --model cfg/yolo.cfg --load ../darknet-resources/yolo.weights --gpu 0.6 --json --recursive
    #command eval: python3 evalpti01.py -p jsonoutput/  -g ~/workspace/datasets/pti/PTI01-bbox-labels/ -i ~/workspace/datasets/pti/PTI01/
    parser.add_argument("--gtpath", help="The path for groundtruths/labels.")
    # parser.add_argument("--prpath", help="The path for groundtruths/labels.") #ptilabelspath
    parser.add_argument("--impath", help="Refers to the dataset image path.")
    parser.add_argument("--implot_seq", type=int, default=None, help="For model yolov2 only. Plot the gt and pred bboxes with available metric in the images from dataset.")
    parser.add_argument("--implot_rand", type=int, default=None, help="Same as --implot_seq, but randomly choose images from dataset for plotting.") #ptiimagespatha

    parser.add_argument('--parse_to_mot_file', action='store_true',
        help='Runs the script for generating the txt files with gt and predicitons in mot format from chainercv data. Requires backup file.')

    args = parser.parse_args()

    if args.parse_to_mot_file:
        if not args.loadfrom:
            parser.error("--parse_to_mot_file requires the backup file defined by --loadfrom argument.")
            return

        parser = Parser()
        '''
        The dataset_name and model_name defined in the parameters below are statical to avoid developing
        extra verifications about compatibility. I know this dataset and this model are ok for this test.
        '''

        parser.convert_chainerdata_to_motdata_files(
            loadfrom=args.loadfrom, dataset_name='PTI01', model_name=args.model.replace('_',''))

        print("Done.")
        return

    if args.implot_seq or args.implot_rand:
        if not args.model == 'yolov2':
            parser.error('Bbox plotting is only available for the model yolov2')
            return
        if not args.impath:
            parser.error('Cannot plot bboxes if there are not impath defined.')
            return

    if args.implot_seq and args.implot_rand:
        print("Both implot_seq args.implot_rand were specified. Ignoring implot_rand.")
        args.implot_rand = None

    if args.plot != None:
        if args.plot == []:
            args.plot = ['all']
        for pl in args.plot:
            if not pl in metric_plot_choices:
                parser.error('The plotting you requested does not exist: {}.'.format(pl))
                return

    if args.model == 'yolov2':
        if not args.loadfrom.endswith('.npy'):
            if not args.gtpath or not args.loadfrom:
                parser.error('Missing --gtpath or --loadfrom that are mandatory for yolov2.')
                return

        run_yolov2(args)
    elif args.model in ['faster_rcnn', 'ssd300', 'ssd512']:
        run_chainercv(args)
    else:
        parser.error('Invalid Model.')
        return


def run_yolov2(args):

    implot = {'rand': False, 'count': 0}
    if args.implot_seq and args.implot_seq > 0:
        imgplot['count'] = args.implot_seq
    elif args.implot_rand and args.implot_seq > 0:
        imgplot['rand'] = True
        imgplot['count'] = args.implot_seq
    else:
        implot = None

    flow_eval = DarkflowEvalPTI01(args.gtpath, args.loadfrom, args.impath, metric=args.metric, limit=args.limit, plot=args.plot, implot=implot)
    flow_eval.eval()

def run_chainercv(args):
    chainercv_eval = ChainercvEvalPTI01(
        args.gtpath, args.impath, args.pretrained_model,
        args.model, args.limit,  args.gpu, args.batchsize, args.metric, args.loadfrom, plot=args.plot)
    chainercv_eval.eval()

if __name__ == '__main__':
    main()
