from math import sqrt

import torch
import torch.nn as nn
import torch.nn.init as init

from .layers import (
    YOLOLayer, ConvBlock, DarknetBlock, SPPLayer,
    BufferLayer, ConcatenateLayer, RouteLayer, ManipulationLayer)


class Basic(nn.Module):
    """
    The Basic module implements a forward function which supporting
    manipulations defined in `layers.py`. See following models for the example
    of porting official models to pytorch.
    """
    def forward(self, x, targets=None):
        """
        Forward path of YOLO.
        Args:
            x (torch.Tensor) : input data whose shape is `(N, C, H, W)`,
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is
                `(M, 6)`, where M is the total number of boxes in this batch.
                The order of last dimension is
                `(batch_id, class_id, x, y, w, h)`.

        Returns:
            if target is not None, then returns:
                loss_dit (dict of torch.Tensor): summation of each loss term
                    for backpropagation, which includes following kyes:
                        `('loss/xy', 'loss/wh', 'loss/obj', 'loss/cls')`.
            otherwise:
                output (torch.Tensor): concatenated detection results of shape
                    [B, N x n_anchors, 6], where B is batchsize, N is the
                    total number of anchors including different grids and
                    scales. The last dimension is in order
                    `(batch_id, class_id, x, y, w, h)`.
        """
        buffer = {}
        for module in self.blocks:
            if isinstance(module, ManipulationLayer):
                x, buffer = module(x, buffer, targets)
            else:
                x = module(x)
        if targets is None:
            pred = torch.cat(buffer['pred'], dim=1)
            return pred
        else:
            loss_dict = {
                k: v for k, v in buffer.items() if k.startswith('loss/')}
            return loss_dict

    def weight_init(self):
        # the default batchnorm initialization is same as darknet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _, c, k, _ = m.weight.shape
                scale = sqrt(2 / (k * k * c))
                init.normal_(m.weight, std=scale)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.zeros_(m.bias)


class YOLOv3(Basic):
    def __init__(self, n_classes, ignore_threshold=0.7):
        """
        Initialization of yolov3.
        Args:
            ignore_threshold (float): The ignore threshold for positive
                objectness loss.
        """
        super().__init__()

        anchors = torch.tensor([
            [116, 90], [156, 198], [373, 326],
            [30, 61],  [62, 45],   [59, 119],
            [10, 13],  [16, 30],   [33, 23],
        ]).float()
        mask = torch.tensor([
            [0, 1, 2],      # dowmsacle 32
            [3, 4, 5],      # downsacle 16
            [6, 7, 8],      # downsacle 8
        ]).long()

        self.blocks = nn.ModuleList([
            # darknet53
            ConvBlock(3, 32, kernel_size=3, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            DarknetBlock(in_channels=64, num_blocks=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            DarknetBlock(in_channels=128, num_blocks=2),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            DarknetBlock(in_channels=256, num_blocks=8),
            BufferLayer("small-1"),
            ConvBlock(256, 512, kernel_size=3, stride=2),
            DarknetBlock(in_channels=512, num_blocks=8),
            BufferLayer("medium-1"),
            ConvBlock(512, 1024, kernel_size=3, stride=2),
            DarknetBlock(in_channels=1024, num_blocks=4),

            # big objects
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            # SPP
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            BufferLayer("medium-2"),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=1024,
                all_anchors=anchors,
                anchors_idxs=mask[0],
                stride=32,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),

            # medium objects
            RouteLayer('medium-2'),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BufferLayer("medium-3"),
            ConcatenateLayer("medium-3", "medium-1"),
            ConvBlock(768, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            BufferLayer("small-2"),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=512,
                all_anchors=anchors,
                anchors_idxs=mask[1],
                stride=16,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),

            # small objects
            RouteLayer('small-2'),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BufferLayer('small-3'),
            ConcatenateLayer("small-3", "small-1"),
            ConvBlock(384, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=256,
                all_anchors=anchors,
                anchors_idxs=mask[2],
                stride=8,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),
        ])

        self.weight_init()


class YOLOv3SPP(Basic):
    def __init__(self, n_classes, ignore_threshold=0.7):
        """
        Initialization of yolov3-spp.
        Args:
            ignore_threshold (float): The ignore threshold for positive
                objectness loss.
        """
        super().__init__()

        anchors = torch.tensor([
            [116, 90], [156, 198], [373, 326],
            [30, 61],  [62, 45],   [59, 119],
            [10, 13],  [16, 30],   [33, 23],
        ]).float()
        mask = torch.tensor([
            [0, 1, 2],      # dowmsacle 32
            [3, 4, 5],      # downsacle 16
            [6, 7, 8],      # downsacle 8
        ]).long()

        self.blocks = nn.ModuleList([
            # darknet53
            ConvBlock(3, 32, kernel_size=3, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            DarknetBlock(in_channels=64, num_blocks=1),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            DarknetBlock(in_channels=128, num_blocks=2),
            ConvBlock(128, 256, kernel_size=3, stride=2),
            DarknetBlock(in_channels=256, num_blocks=8),
            BufferLayer("small-1"),
            ConvBlock(256, 512, kernel_size=3, stride=2),
            DarknetBlock(in_channels=512, num_blocks=8),
            BufferLayer("medium-1"),
            ConvBlock(512, 1024, kernel_size=3, stride=2),
            DarknetBlock(in_channels=1024, num_blocks=4),

            # big objects
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            SPPLayer([13, 9, 5, 1]),
            ConvBlock(2048, 512, kernel_size=1, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1),
            BufferLayer("medium-2"),
            ConvBlock(512, 1024, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=1024,
                all_anchors=anchors,
                anchors_idxs=mask[0],
                stride=32,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),

            # medium objects
            RouteLayer('medium-2'),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BufferLayer("medium-3"),
            ConcatenateLayer("medium-3", "medium-1"),
            ConvBlock(768, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            ConvBlock(512, 256, kernel_size=1, stride=1),
            BufferLayer("small-2"),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=512,
                all_anchors=anchors,
                anchors_idxs=mask[1],
                stride=16,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),

            # small objects
            RouteLayer('small-2'),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BufferLayer('small-3'),
            ConcatenateLayer("small-3", "small-1"),
            ConvBlock(384, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=256,
                all_anchors=anchors,
                anchors_idxs=mask[2],
                stride=8,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),
        ])

        self.weight_init()


class TinyYOLOv3(Basic):
    def __init__(self, n_classes, ignore_threshold=0.7):
        """
        Initialization of yolov3-tiny.
        Args:
            ignore_threshold (float): The ignore threshold for positive
                objectness loss.
        """
        super().__init__()

        anchors = torch.tensor([
            [81, 82], [135, 169], [344, 319],
            # correct anchor
            # [10, 14], [23, 27], [37, 58],
            # pretrained anchor
            [23, 27], [37, 58], [81, 82],
        ]).float()
        mask = torch.tensor([
            [0, 1, 2],      # dowmsacle 32
            [3, 4, 5],      # downsacle 16
        ]).long()

        self.blocks = nn.ModuleList([
            # tiny-yolo backbone
            ConvBlock(3, 16, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(16, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, kernel_size=3, stride=1),
            BufferLayer('medium-1'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1),

            # big objects
            ConvBlock(1024, 256, kernel_size=1, stride=1),
            BufferLayer("medium-2"),
            ConvBlock(256, 512, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=512,
                all_anchors=anchors,
                anchors_idxs=mask[0],
                stride=32,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),

            # medium objects
            RouteLayer('medium-2'),
            ConvBlock(256, 128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BufferLayer("medium-3"),
            ConcatenateLayer("medium-3", "medium-1"),
            ConvBlock(384, 256, kernel_size=3, stride=1),
            YOLOLayer(
                in_channels=256,
                all_anchors=anchors,
                anchors_idxs=mask[1],
                stride=16,
                n_classes=n_classes,
                ignore_threshold=ignore_threshold),
        ])

        self.weight_init()


YOLOs = {
    'yolov3': YOLOv3,
    'yolov3-spp': YOLOv3SPP,
    'yolov3-tiny': TinyYOLOv3,
}
