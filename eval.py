import argparse

import torch
from pycocotools.coco import COCO

from yolov3.dataset import DetectionDataset
from yolov3.transforms import preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', default='yolov3', choices=YOLOs.keys(),
                    help='model name')
parser.add_argument('--weights', type=str, required=True,
                    help='path to weights file')
parser.add_argument('--n_classes', default=80, type=int,
                    help='nunmber of classes')
parser.add_argument('--conf_threshold', type=float, default=0.005,
                    help='confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.45,
                    help='nms threshold')
parser.add_argument('--img_size', type=int, default=416,
                    help='evaluation image size')
parser.add_argument('--val_batch_size', default=32, type=int,
                    help='evaluation batch size')
parser.add_argument('--num_workers', default=6, type=int,
                    help='dataloader workers')
parser.add_argument('--val_ann_file', type=str,
                    default='./data/coco/annotations/instances_5k.json',
                    help='path to val annotation file')
parser.add_argument('--val_img_root', type=str,
                    default='./data/coco/all2014',
                    help='path to root of val images')
args = parser.parse_args()


def main():
    # Initiate model
    model = YOLOs[args.model](args.n_classes).to(device)
    if args.weights:
        if args.weights.endswith('.pt'):
            print("loading pytorch weights:", args.weights)
            ckpt = torch.load(args.weights)
            model.load_state_dict(ckpt['model'])
        else:
            print("loading darknet weights:", args.weights)
            parse_weights(model, args.weights)
    model.eval()

    preprocess.update_img_size(args.img_size)
    loader = torch.utils.data.DataLoader(
        dataset=DetectionDataset(
            COCO(args.val_ann_file),
            args.val_img_root,
            args.img_size,
            transforms=preprocess),
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        collate_fn=DetectionDataset.collate,
        shuffle=False,
        drop_last=False)

    _, _ = evaluate(
        model, loader, args.conf_threshold, args.nms_threshold, device)


if __name__ == '__main__':
    main()
