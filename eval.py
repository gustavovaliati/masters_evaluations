import argparse
from myevaluations.darkflow.evalpti01_darkflow import DarkflowEvalPTI01
from myevaluations.chainercv.evalpti01_chainercv import ChainercvEvalPTI01

#python3 eval.py --model yolov2 --loadfrom ~/workspace/darkflow/jsonoutput/  --gtpath ~/workspace/datasets/pti/PTI01-bbox-labels/
#python3 eval.py --model ssd512 --gpu 0 --impath ~/workspace/datasets/pti/PTI01/ --gtpath ~/workspace/datasets/pti/PTI01-bbox-labels/

def main():
    parser = argparse.ArgumentParser(description="An evaluator for the masters degree experiments.")
    parser.add_argument(
        '--model', choices=('faster_rcnn', 'ssd300', 'ssd512', 'yolov2'),
        default=False, required = True)
    parser.add_argument(
        '--metric', choices=('voc_detection','mot','all'),
        default='all')

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
    parser.add_argument("--impath", help="chainercv argument") #ptiimagespath
    args = parser.parse_args()

    if args.model == 'yolov2':
        if not args.gtpath or not args.loadfrom:
            parser.error('Missing --gtpath or --loadfrom that are mandatory for yolov2.')
            return

        run_yolov2(args)
    elif args.model in ['faster_rcnn', 'ssd300', 'ssd512']:
        run_chainercv(args)
    else:
        raise Exception('Invalid Model.')


def run_yolov2(args):
    flow_eval = DarkflowEvalPTI01(args.gtpath, args.loadfrom, args.impath, metric=args.metric, limit=args.limit)
    flow_eval.eval()

def run_chainercv(args):
    chainercv_eval = ChainercvEvalPTI01(
        args.gtpath, args.impath, args.pretrained_model,
        args.model, args.limit,  args.gpu, args.batchsize, args.metric, args.loadfrom)
    chainercv_eval.eval()

if __name__ == '__main__':
    main()
