import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from PIL import Image


from .transforms import ToTensor


class DetectionDataset(Dataset):
    def __init__(self,
                 coco,
                 img_root,
                 img_size=416,
                 transforms=ToTensor()):
        self.coco = coco
        self.img_root = img_root
        self.img_size = img_size
        self.transforms = transforms
        self.ids = self.coco.getImgIds()

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.label2categoryid = [cat['id'] for cat in cats]
        self.label2name = [cat['name'] for cat in cats]
        self.label2color = torch.randint(
            0, 256, size=(len(self.label2name), 3),
            generator=torch.Generator().manual_seed(1))
        self.categoryid2label = {
            k: v for v, k in enumerate(self.label2categoryid)}
        self.n_classes = len(self.label2name)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        if type(self.img_size) == int:
            img_size = self.img_size
        else:
            img_size = self.img_size[index]

        # load image and preprocess
        img_dict = self.coco.loadImgs(id)[0]
        img_path = os.path.join(
            self.img_root, os.path.basename(img_dict['file_name']))
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        orig_size = torch.tensor([orig_h, orig_w]).long()

        # load annotations from cocoapi
        anno_ids = self.coco.getAnnIds(imgIds=id)
        annotations = self.coco.loadAnns(anno_ids)
        labels = []
        bboxes = []
        for anno in annotations:
            x, y, w, h = anno['bbox']
            label = self.categoryid2label[anno['category_id']]
            if w > 1 and h > 1:
                labels.append(label)
                bboxes.append([x + w / 2, y + h / 2, w, h])
        labels = torch.tensor(labels).float()
        bboxes = torch.tensor(bboxes).float()
        if len(bboxes) == 0:
            bboxes = torch.empty((0, 4)).float()
        self.transforms.update_img_size(img_size)
        img, bboxes = self.transforms(img, bboxes)
        # to [0, 1]
        bboxes /= img_size
        # [cls, x, y, w, h]
        targets = torch.cat([labels[:, None], bboxes], dim=1)

        return img, targets, orig_size, torch.tensor(id).long()

    @staticmethod
    def collate(batch):
        b_img, b_targets, b_orig_size, b_id = zip(*batch)
        b_targets = list(b_targets)
        for batch_idx, target in enumerate(b_targets):
            b_targets[batch_idx] = F.pad(target, (1, 0), value=batch_idx)
        return (
            torch.stack(b_img, dim=0),
            torch.cat(b_targets, dim=0),
            torch.stack(b_orig_size, dim=0),
            torch.stack(b_id, dim=0),
        )


class DistributedMultiScaleSampler(Sampler):
    """Combine distributed smapler with multi-scale training.

    This module implement multi-scale training by randomly initializing the
    size of images at the begining of each epoch.
    """
    def __init__(self, dataset, batch_size,
                 scale_interval, scale_range=(320, 608),
                 seed=0):
        self.dataset = dataset
        self.seed = seed
        self.scale_interval = batch_size * scale_interval
        # The actual length of dataset when considering the scale_interval and
        # distributed training at the same time. The length guarantee that the
        # number of batchs for each random image size is the same.
        self.length = (
            len(self.dataset) // self.scale_interval * self.scale_interval)
        self.low = scale_range[0] // 32
        self.high = scale_range[1] // 32
        self.rank = dist.get_rank()
        self.num_replicas = dist.get_world_size()

    def set_epoch(self, epoch):
        # use random generator to keep consistent across distributed processes
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        img_sizes = torch.randint(
            self.low, self.high + 1, (self.length // self.scale_interval,),
            generator=g)
        img_sizes *= 32
        img_sizes = torch.repeat_interleave(img_sizes, self.scale_interval)
        # the `img_size` looks like
        # [320, 320, 320, 320, 418, 418, 418, 418, 608, 608, 608, 608, ...]

        indices = torch.randperm(
            len(self.dataset), generator=g)[:self.length]
        reordered_sizes = torch.zeros(len(self.dataset)).long()
        reordered_sizes[indices] = img_sizes

        self.dataset.img_size = reordered_sizes
        self.indices = indices[self.rank:self.length:self.num_replicas]
        # `self.indices` only contains partial indices used by process
        assert len(self.indices) == self.length // self.num_replicas

    def __iter__(self):
        if not hasattr(self, 'indices'):
            raise RuntimeError('call set_epoch() first')
        return iter(self.indices)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    from pycocotools.coco import COCO
    from torchvision.transforms.functional import to_pil_image

    from yolov3.utils import draw_bbox, draw_text
    from yolov3.transforms import preprocess, augmentation

    img_size = 416
    ann_file = './data/coco/annotations/instances_5k.json'
    img_root = './data/coco/all2014/'
    coco = COCO(ann_file)

    # unittesting for test set
    dataset = DetectionDataset(coco, img_root, img_size, preprocess)
    img, targets, orig_size, id = dataset[2]
    labels, bboxes = torch.split(targets, [1, 4], dim=1)
    bboxes *= img_size

    # forward test
    target_img = to_pil_image(img)
    for bbox, label in zip(bboxes, labels):
        name = dataset.label2name[int(label)]
        color = dataset.label2color[int(label)]
        draw_bbox(target_img, bbox, name, color)
        draw_text(target_img, bbox, name, color)
    target_img.save('test1.png')

    # revert test
    file_path = coco.loadImgs(int(id))[0]['file_name']
    bboxes = preprocess.revert(bboxes, orig_size, img_size)
    target_img = Image.open(os.path.join(img_root, file_path))
    for bbox, label in zip(bboxes, labels):
        name = dataset.label2name[int(label)]
        color = dataset.label2color[int(label)]
        draw_bbox(target_img, bbox, name, color)
        draw_text(target_img, bbox, name, color)
    target_img.save('test2.png')

    new_img_size = 608
    dataset.img_size = new_img_size
    img, targets, orig_size, id = dataset[2]
    labels, bboxes = torch.split(targets, [1, 4], dim=1)
    bboxes *= new_img_size

    target_img = to_pil_image(img)
    for bbox, label in zip(bboxes, labels):
        name = dataset.label2name[int(label)]
        color = dataset.label2color[int(label)]
        draw_bbox(target_img, bbox, name, color)
        draw_text(target_img, bbox, name, color)
    target_img.save('test3.png')

    # unittesting for train set
    dataset = DetectionDataset(coco, img_root, img_size, augmentation)
    img, targets, orig_size, id = dataset[2]
    labels, bboxes = torch.split(targets, [1, 4], dim=1)
    bboxes *= img_size

    # forward test
    target_img = to_pil_image(img)
    for bbox, label in zip(bboxes, labels):
        name = dataset.label2name[int(label)]
        color = dataset.label2color[int(label)]
        draw_bbox(target_img, bbox, name, color)
        draw_text(target_img, bbox, name, color)
    target_img.save('test4.png')
