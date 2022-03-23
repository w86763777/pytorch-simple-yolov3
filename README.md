# YOLOv3 in PyTorch

[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)

A simple pytorch loop-less implementation of yolo loss and non-maximum
suppression (NMS) based on [`torchvision.ops`](https://pytorch.org/vision/stable/ops.html).

In this implementation, the most details follow the darknet implementation
except for:
- The MSE loss of bbox (`x`, `y`) is replaced with binary cross entropy.
- The parameter `subdivision` is replaced with `accumulation`. The effective
batch size in this implementation is `batch_size` x `accumulation`.


## Environments
Install all required packages:
```bash
$ pip install -r requirements.txt
```

## Pretraind Darknet Weights
- Download pretrained backbone and 3 full models weights.
  ```bash
  $ cd weights
  $ sh download_weights.sh
  ```

## MS COCO 2014
- The script will download [MS COCO 2014](https://cocodataset.org/#download)
and rearrange them into the trainval splits like darknet.
  ```bash
  $ cd data
  $ sh download_coco.sh
  ```
  There is a python script which will be executed at the end of
  `download_coco.sh`, it needs `pycocotool` to be installed in advance. If the installation is missed, it was nothing serious. You can manually run `python generate_annotations.py` inside the directory `data` to generate darknet annotations in coco format.

## Evalutate Darknet Weights on MS COCO 2014

Use `--model` and `--weights` to set model architecture and pretrained weights.
```bash
python eval.py --model yolov3 --weights ./weights/yolov3.weights --img_size 416
```

|Model      |AP@.5(darknet)|AP@.5 (our)|`--img_size`|
|-----------|--------------|-----------|------------|
|yolov3-320 |51.5          |51.4       |320         |
|yolov3-416 |55.3          |55.3       |416         |
|yolov3-608 |57.9          |58.4       |608         |
|yolov3-tiny|33.1          |32.8       |416         |
|yolov3-spp |60.6          |61.0       |608         |

**NOTE**, the darkent weights for `yolov3-tiny` was trained with incorrect prior anchors which is listed
[here](https://github.com/w86763777/pytorch-simple-yolov3/blob/master/yolov3/models/__init__.py#L280).
The correct one is showed beside it for the reference.

## Inference on Single Image
```bash
python demo.py --image ./data/street.jpg --model yolov3 --weights ./weights/yolov3.weights --img_size 418
```
The confidence threshold and nms IoU threshold can be changed by `--conf_threshold`
and `--nms_threshold` respectively.

## Train From Scratch

The default training arguments are same as official ones for `yolov3`.

- Single GPU training (default to `cuda:0`)
  ```bash
  python train.py \
    --weights ./wegiths/darknet53.conv.74 \
    --logdir ./logs/yolov3
  ```
  Default `batch_size=4` and `accumulation=16` are designed for single GPU with
  8G VRAM.

- Multi GPU training
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --weights ./wegiths/darknet53.conv.74 \
    --logdir ./logs/yolov3 \
    --batch_size 64 \
    --accumulation 1
  ```
  The inter-process communication uses port `39846` by default. It can be
  changed by editing `train.py`.

### Results on MS COCO 2014 (darknet splits)

|Model     |AP@.5(darknet)|AP@.5(our)|AP@.5:.95(our)|
|----------|--------------|----------|--------------|
|yolov3-320|51.5          |48.5      |27.5          |
|yolov3-418|55.3          |53.0      |30.6          |
|yolov3-608|57.9          |54.8      |31.7          |