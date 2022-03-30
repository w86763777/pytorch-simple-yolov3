import argparse

import torch
from PIL import Image

from yolov3.transforms import preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import nms, draw_bbox, draw_text


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=str, required=True)
# yolo
parser.add_argument('--model', default='yolov3', choices=YOLOs.keys(),
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
args = parser.parse_args()


label2name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

label2color = torch.randint(
    0, 256, size=(len(label2name), 3),
    generator=torch.Generator().manual_seed(1))


def main():
    # The class names and colors can be obtained from dataset.
    # from pycocotools.coco import COCO
    # from yolov3.dataset import DetectionDataset
    # dataset = DetectionDataset(
    #     COCO("./data/coco/annotations/instances_5k.json"),
    #     img_root=None,
    #     img_size=None,
    #     transforms=None)
    # label2name = dataset.label2name
    # label2color = dataset.label2color

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
            name = label2name[int(label)]
            color = label2color[int(label)]
            draw_bbox(img, bbox, name, color)
            draw_text(img, bbox, name, color)
            print('+ Label: %s, Conf: %.5f' % (name, score))
    else:
        print("No Objects Deteted!!")
    img.save("demo.png")


if __name__ == '__main__':
    main()
