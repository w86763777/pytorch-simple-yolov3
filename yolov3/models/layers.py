import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # For loading pretrained darknet, the parameter names must be
        # `conv` and `bn` for Convolution and BatchNormalization.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2, 1, 1)
        self.conv2 = ConvBlock(in_channels // 2, in_channels, 3, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = h + x
        return h


class DarknetBlock(nn.Module):
    def __init__(self, in_channels, num_blocks=1):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(in_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class SPPLayer(nn.Module):
    def __init__(self, kernel_sizes):
        super().__init__()
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        x = [pooling(x) for pooling in self.poolings]
        return torch.cat(x, dim=1)


class ManipulationLayer(nn.Module):
    """
    The buffer and targets are engaged in the forward propagation of these
    kinds of layers.
    """
    def forward(self, x, buffer, targets):
        raise NotImplementedError()


class BufferLayer(ManipulationLayer):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x, buffer, targets):
        assert self.name not in buffer
        buffer[self.name] = x

        return x, buffer


class ConcatenateLayer(ManipulationLayer):
    def __init__(self, name1, name2, dim=1):
        super().__init__()
        self.name1 = name1
        self.name2 = name2
        self.dim = 1

    def forward(self, x, buffer, targets):
        x = torch.cat([buffer[self.name1], buffer[self.name2]], dim=self.dim)
        return x, buffer


class RouteLayer(ManipulationLayer):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x, buffer, targets):
        return buffer[self.name], buffer


def sanity_check(x, y, w, h, obj, cls):
    """A finite value check before loss calculation."""
    assert torch.isfinite(x).all(), "Network diverge x: " + str(x)
    assert torch.isfinite(y).all(), "Network diverge y: " + str(y)
    assert torch.isfinite(w).all(), "Network diverge w: " + str(w)
    assert torch.isfinite(h).all(), "Network diverge h: " + str(h)
    assert torch.isfinite(obj).all(), "Network diverge obj: " + str(obj)
    assert torch.isfinite(cls).all(), "Network diverge cls: " + str(cls)


class YOLOLayer(ManipulationLayer):
    def __init__(self,
                 in_channels,
                 all_anchors,
                 anchors_idxs,
                 stride,
                 n_classes,
                 ignore_threshold):
        super().__init__()
        self.anchors_idxs = anchors_idxs
        self.stride = stride
        self.n_classes = n_classes
        self.ignore_threshold = ignore_threshold

        all_anchors = all_anchors.clone() / stride
        self.register_buffer('anchors', all_anchors[self.anchors_idxs])
        self.register_buffer('all_anchors', all_anchors)

        self.bce = nn.BCELoss(reduction='none')
        # For loading pretrained darknet, the parameter name must be `conv` for
        # Convolution.
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=len(anchors_idxs) * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x, buffer, labels):
        out = self.conv(x)
        device = out.device

        B, C, N, N = out.shape      # batch_size, ?, grid, grid
        A = C // (5 + self.n_classes)   # num_anchors

        out = out.view(B, A, 5 + self.n_classes, N, N)
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        out[..., :2] = torch.sigmoid(out[..., :2])
        out[..., 4:] = torch.sigmoid(out[..., 4:])

        with torch.no_grad():
            out_bboxes = out[..., :4].detach().clone()
            # Batch, Anchor, N, N
            x_shift = torch.arange(N).view(1, 1, 1, -1).to(device)
            y_shift = torch.arange(N).view(1, 1, -1, 1).to(device)
            # Batch, Anchor, N, N
            w_anchors = self.anchors[:, 0].view(1, -1, 1, 1)
            h_anchors = self.anchors[:, 1].view(1, -1, 1, 1)
            # grid coordinates
            out_bboxes[..., 0] += x_shift
            out_bboxes[..., 1] += y_shift
            out_bboxes[..., 2] = torch.exp(out_bboxes[..., 2]) * w_anchors
            out_bboxes[..., 3] = torch.exp(out_bboxes[..., 3]) * h_anchors

            # pixel coordinates
            pred = torch.cat([out_bboxes * self.stride, out[..., 4:]], dim=-1)
            pred = pred.view(B, -1, 5 + self.n_classes)
            buffer['pred'] = buffer.get('pred', []) + [pred]

        if labels is None:
            return pred, buffer

        out = out.view(-1, 5 + self.n_classes)
        assert out.shape[0] == B * A * N * N
        out_batchidx = torch.arange(B).repeat_interleave(A * N * N)
        out_batchidx = out_batchidx.to(device)

        with torch.no_grad():
            lbl_batchidx, lbl_cls, lbl_bboxes = torch.split(
                labels, [1, 1, 4], dim=1)
            lbl_batchidx = lbl_batchidx.long().view(-1)
            lbl_cls = lbl_cls.long().view(-1)
            # grid coordinates
            lbl_bboxes = lbl_bboxes * N

            # negative objectness
            out_bboxes = out_bboxes.view(-1, 4)
            out_bboxes[:, [0, 1]] += (out_batchidx * N)[:, None]
            lbl_bboxes[:, [0, 1]] += (lbl_batchidx * N)[:, None]
            out_lbl_IoU = box_iou(
                box_convert(out_bboxes, 'cxcywh', 'xyxy'),
                box_convert(lbl_bboxes, 'cxcywh', 'xyxy'),
            )
            out_lbl_maxIoU = out_lbl_IoU.amax(dim=1)
            lbl_bboxes[:, [0, 1]] -= (lbl_batchidx * N)[:, None]
            # objectness target
            tgt_obj_mask = out_lbl_maxIoU < self.ignore_threshold
            tgt_obj = torch.zeros(len(out)).to(device)

            # box loss, positive objectness loss, class loss
            lbl_anchors_IoU = box_iou(
                F.pad(lbl_bboxes[:, 2:], (2, 0)),
                F.pad(self.all_anchors, (2, 0)))
            tgt_anchors_idx = torch.argmax(lbl_anchors_IoU, dim=1)
            lbl_mask = torch.stack(
                [tgt_anchors_idx == idx for idx in self.anchors_idxs], dim=1)
            lbl_mask = lbl_mask.any(dim=1)
            lbl_cls = lbl_cls[lbl_mask]
            lbl_bboxes = lbl_bboxes[lbl_mask]
            lbl_batchidx = lbl_batchidx[lbl_mask]
            tgt_anchors_idx = tgt_anchors_idx[lbl_mask]
            tgt_anchors_idx %= len(self.anchors_idxs)

            lbl_x, lbl_y, lbl_w, lbl_h = lbl_bboxes.T
            lbl_i = lbl_y.floor().long()
            lbl_j = lbl_x.floor().long()

            lbl_idx2out_idx = (
                N * N * A * lbl_batchidx +
                N * N * tgt_anchors_idx +
                N * lbl_i +
                lbl_j)

            # box target
            tgt_x = lbl_x - lbl_x.floor()
            tgt_y = lbl_y - lbl_y.floor()
            tgt_w = torch.log(lbl_w / self.anchors[tgt_anchors_idx, 0] + 1e-16)
            tgt_h = torch.log(lbl_h / self.anchors[tgt_anchors_idx, 1] + 1e-16)
            bbox_scale = 2 - (lbl_w / N) * (lbl_h / N)
            # objectness target
            tgt_obj_mask[lbl_idx2out_idx] = True
            tgt_obj[lbl_idx2out_idx] = 1
            # class target
            tgt_cls = torch.eye(self.n_classes)[lbl_cls].to(device)

        sanity_check(
            out[:, 0], out[:, 1], out[:, 2], out[:, 4], out[:, 4], out[:, 5:])

        loss_obj = (tgt_obj_mask * self.bce(out[:, 4], tgt_obj)).sum()
        out = out[lbl_idx2out_idx]
        loss_cls = self.bce(out[:, 5:], tgt_cls).sum()
        # dividing by 2 to match darkent implementation
        loss_x = (bbox_scale * self.bce(out[:, 0], tgt_x)).sum()
        loss_y = (bbox_scale * self.bce(out[:, 1], tgt_y)).sum()
        loss_w = (bbox_scale * (out[:, 2] - tgt_w) ** 2 / 2).sum()
        loss_h = (bbox_scale * (out[:, 3] - tgt_h) ** 2 / 2).sum()

        buffer['loss/xy'] = buffer.get('loss/xy', 0) + loss_x + loss_y
        buffer['loss/wh'] = buffer.get('loss/wh', 0) + loss_w + loss_h
        buffer['loss/obj'] = buffer.get('loss/obj', 0) + loss_obj
        buffer['loss/cls'] = buffer.get('loss/cls', 0) + loss_cls

        return pred, buffer
