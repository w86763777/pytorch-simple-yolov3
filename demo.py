import argparse

import torch
from pycocotools.coco import COCO
from PIL import Image

from yolov3.dataset import DetectionDataset
from yolov3.transforms import preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import nms, draw_bbox, draw_text


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=str, required=True)
# yolo
parser.add_argument('--model', default='yolo', choices=YOLOs.keys(),
                    help='model name')
parser.add_argument('--weights', type=str, required=True,
                    help='path to weights file')
parser.add_argument('--n_classes', default=80, type=int,
                    help='nunmber of classes')
# demo
parser.add_argument('--img_size', type=int, default=416,
                    help='evaluation image size')
parser.add_argument('--conf_threshold', type=float, default=0.5,
                    help='confidence threshold')
parser.add_argument('--nms_threshold', type=float, default=0.45,
                    help='nms threshold')
# for class label
parser.add_argument('--val_ann_file', type=str,
                    default='./data/coco/annotations/instances_5k.json',
                    help='path to val annotation file')
args = parser.parse_args()


def main():
    # the dataset is only used for getting class names and colors
    dataset = DetectionDataset(
        COCO(args.val_ann_file),
        img_root=None,
        img_size=None,
        transforms=None)

    img = Image.open(args.image).convert('RGB')
    orig_size = torch.tensor([img.size[1], img.size[0]]).long()
    img, _ = preprocess(img)
    img = img.unsqueeze(0)

    # Initiate model
    model = YOLOs[args.model](args.n_classes).to(device)
    if args.weights.endswith('.pt'):
        ckpt = torch.load(args.weights)
        model.load_state_dict(ckpt['model'])
    else:
        parse_weights(model, args.weights)
    model.eval()

    with torch.no_grad():
        img, orig_size = img.to(device), orig_size.to(device)
        bboxes = model(img)
        bboxes = nms(bboxes, args.conf_threshold, args.nms_threshold)[0]

    img = Image.open(args.image)
    if len(bboxes):
        bboxes, scores, labels = torch.split(bboxes, [4, 1, 1], dim=1)
        bboxes = preprocess.revert(bboxes, orig_size, args.img_size)
        for bbox, score, label in zip(bboxes, scores, labels):
            name = dataset.label2name[int(label)]
            color = dataset.label2color[int(label)]
            draw_bbox(img, bbox, name, color)
            draw_text(img, bbox, name, color)
            print('+ Label: %s, Conf: %.5f' % (name, score))
    else:
        print("No Objects Deteted!!")
    img.save("demo.png")


if __name__ == '__main__':
    main()
