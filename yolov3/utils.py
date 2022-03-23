import torch
from PIL import ImageDraw, ImageFont
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms, box_convert
from tqdm import tqdm


def nms(predictions, conf_thre=0.7, nms_iou_threshold=0.45):
    """
    The implementation of batched non-maximum suppression algorithm based on
    torchvision.ops.batched_nms.

    Args:
        predictions (torch.tensor): The raw output of yolo models. The unit of
            bbox is in pixel.
        conf_thre (float): confidence threshold of bounding boxes used in first
            stage filtering.
        nms_iou_threshold (float): IoU threshold used in
            `torchvision.ops.batched_nms`.
    Returns:
        outputs (list of torch.tensor): A list of tensors with length equals to
            `predictions`. Each tensor is the nms results of corresponding
            tensor of `predictions`.
    """
    B, K, D = predictions.shape
    n_classes = D - 5
    batch_indices = torch.arange(
        0, B).repeat_interleave(K).to(predictions.device)
    predictions = predictions.view(B * K, -1)

    # Filter out confidence scores below threshold
    conf_mask = predictions[:, 5:] * predictions[:, 4][:, None] >= conf_thre
    conf_index, labels = conf_mask.nonzero().t()
    batch_indices = batch_indices[conf_index]
    bboxes = predictions[conf_index, :4]
    scores = predictions[conf_index, 4] * predictions[conf_index, 5 + labels]

    # shift labels to separate same class_id between different images
    idxs = batch_indices * n_classes + labels
    # apply pytorch nms
    selected = batched_nms(
        box_convert(bboxes, 'cxcywh', 'xyxy'), scores, idxs, nms_iou_threshold)

    predictions = torch.cat(
        [bboxes[selected], scores[selected, None], labels[selected, None]],
        dim=1)
    batch_indices = batch_indices[selected]
    outputs = []
    for batch_index in range(B):
        outputs.append(predictions[batch_indices == batch_index])

    return outputs


def draw_bbox(img, bbox, name=None, color="white"):
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    x1, y1 = (x - w / 2).long().item(), (y - h / 2).long().item()
    x2, y2 = (x + w / 2).long().item(), (y + h / 2).long().item()
    draw.rectangle([x1 - 1, y1 - 1, x2 - 1, y2 - 1], width=1, outline='black')
    draw.rectangle([x1 - 1, y1 + 1, x2 - 1, y2 + 1], width=1, outline='black')
    draw.rectangle([x1 + 1, y1 - 1, x2 + 1, y2 - 1], width=1, outline='black')
    draw.rectangle([x1 + 1, y1 + 1, x2 + 1, y2 + 1], width=1, outline='black')
    draw.rectangle([x1, y1, x2, y2], width=1, outline=tuple(color.numpy()))


def draw_text(img, bbox, name, color="white", font_size=16):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./data/arial.ttf", font_size)
    x, y, w, h = bbox
    x, y = (x - w / 2).long().item(), (y - h / 2).long().item() - font_size - 2
    draw.text((x - 1, y - 1), name, font=font, fill='black')
    draw.text((x + 1, y - 1), name, font=font, fill='black')
    draw.text((x - 1, y + 1), name, font=font, fill='black')
    draw.text((x + 1, y + 1), name, font=font, fill='black')
    draw.text((x, y), name, font=font, fill=tuple(color.numpy()))


def evaluate(model, loader, conf_threshold, nms_threshold, device):
    """Evaluate detection model by pycocotools.

    Args:
        model: detection model
        loader (torch DataLoader): the dataset must be `DetectionDataset`.
        conf_threshold: confidence threshold of bounding boxes
        nms_threshold: non-maximum suppression threshold
    Returns:
        AP@0.5:0.95 (float): calculated COCO AP for IoU=0.50:0.95
        AP@0.5 (float): calculated COCO AP for IoU=0.50
    """
    dataset = loader.dataset
    img_size = dataset.img_size
    cocoDt = []
    for imgs, _, orig_sizes, ids in tqdm(loader, leave=False, ncols=0):
        with torch.no_grad():
            imgs, orig_sizes = imgs.to(device), orig_sizes.to(device)
            predictions = model(imgs)
            predictions = nms(predictions, conf_threshold, nms_threshold)

        for id, prediction, orig_size in zip(ids, predictions, orig_sizes):
            if len(prediction) == 0:
                continue
            bboxes, scores, labels = torch.split(prediction, [4, 1, 1], dim=1)
            bboxes = dataset.transforms.revert(bboxes, orig_size, img_size)
            for bbox, score, label in zip(bboxes, scores, labels):
                category_id = dataset.label2categoryid[int(label)]
                # the xy in COCO fomat is the upper left corner
                bbox = [
                    bbox[0].item() - bbox[2].item() / 2,
                    bbox[1].item() - bbox[3].item() / 2,
                    bbox[2].item(),
                    bbox[3].item(),
                ]
                # COCO json format
                Dt = {
                    "image_id": id.item(),
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score.item(),
                    "segmentation": []
                }
                cocoDt.append(Dt)

    if len(cocoDt) > 0:
        cocoGt = dataset.coco
        cocoDt = cocoGt.loadRes(cocoDt)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ap50, ap50_95 = cocoEval.stats[0], cocoEval.stats[1]
    else:
        ap50, ap50_95 = 0, 0
    return ap50, ap50_95
