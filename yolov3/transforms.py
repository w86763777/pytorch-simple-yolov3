import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_tensor, resize, pad, hflip


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=torch.empty((0, 4))):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes

    def revert(self, bboxes, orig_size, img_size):
        for t in reversed(self.transforms):
            if hasattr(t, 'revert'):
                bboxes = t.revert(bboxes, orig_size, img_size)
        return bboxes

    def update_img_size(self, img_size):
        for t in self.transforms:
            if hasattr(t, 'update_img_size'):
                t.update_img_size(img_size)


class ToTensor(object):
    def __call__(self, img, bboxes):
        return to_tensor(img), bboxes

    def revert(self, bboxes, orig_size, img_size):
        return bboxes


class RandomColorJitter(ColorJitter):
    def __init__(self, brightness=0.4, saturation=0.4, hue=0.1):
        super().__init__(
            brightness=brightness, saturation=saturation, hue=hue)

    def __call__(self, img, bboxes):
        img = super().__call__(img)
        return img, bboxes


class RandomResize(object):
    def __init__(self, img_size, scale=(0.7, 1.3)):
        self.img_size = img_size
        self.scale = scale

    def __call__(self, img, bboxes):
        w, h = img.size
        pesudo_h = torch.randint(
            int(h * self.scale[0]), int(h * self.scale[1]) + 1, ())
        pesudo_w = torch.randint(
            int(w * self.scale[0]), int(w * self.scale[1]) + 1, ())
        ratio = pesudo_h / pesudo_w
        if ratio > 1:
            new_h = self.img_size
            new_w = self.img_size / ratio
        else:
            new_h = self.img_size * ratio
            new_w = self.img_size

        img = resize(img, (int(new_h), int(new_w)))
        bboxes[:, [0, 2]] *= new_w / w
        bboxes[:, [1, 3]] *= new_h / h
        return img, bboxes

    def update_img_size(self, img_size):
        self.img_size = img_size


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        w, h = img.size
        if torch.rand(1) < self.p:
            img = hflip(img)
            bboxes[:, 0] = w - bboxes[:, 0]
        return img, bboxes


class Resize(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img, bboxes):
        w, h = img.size
        if h > w:
            ratio = self.img_size / h
            new_h = self.img_size
            new_w = w * ratio
        else:
            ratio = self.img_size / w
            new_h = h * ratio
            new_w = self.img_size
        img = resize(img, (int(new_h), int(new_w)))
        bboxes *= ratio
        return img, bboxes

    def revert(self, bboxes, orig_size, img_size):
        orig_h, orig_w = orig_size
        if orig_h > orig_w:
            scale = orig_h / self.img_size
        else:
            scale = orig_w / self.img_size
        bboxes *= scale
        return bboxes

    def update_img_size(self, img_size):
        self.img_size = img_size


class RandomSquarePad(object):
    def __call__(self, img, bboxes):
        w, h = img.size
        if h > w:
            diff = h - w
            pad_l = torch.randint(0, diff + 1, ())
            pad_r = diff - pad_l
            padding = [pad_l, 0, pad_r, 0]
        else:
            diff = w - h
            pad_t = torch.randint(0, diff + 1, ())
            pad_b = diff - pad_t
            padding = [0, pad_t, 0, pad_b]
        # left top right bottom
        img = pad(img, padding, padding_mode='constant', fill=127)
        bboxes[:, 0] += padding[0]
        bboxes[:, 1] += padding[1]
        return img, bboxes


class SquarePad(object):
    def __call__(self, img, bboxes):
        w, h = img.size
        if h > w:
            diff = h - w
            pad_l = diff // 2
            pad_r = diff - pad_l
            padding = [pad_l, 0, pad_r, 0]
        else:
            diff = w - h
            pad_t = diff // 2
            pad_b = diff - pad_t
            padding = [0, pad_t, 0, pad_b]
        # left top right bottom
        img = pad(img, padding, padding_mode='constant', fill=127)
        bboxes[:, 0] += padding[0]
        bboxes[:, 1] += padding[1]
        return img, bboxes

    def revert(self, bboxes, orig_size, img_size):
        orig_h, orig_w = orig_size
        if orig_h > orig_w:
            diff = img_size - (orig_w * img_size / orig_h).long()
            pad_l = torch.floor(diff / 2)
            pad_t = 0
        else:
            diff = img_size - (orig_h * img_size / orig_w).long()
            pad_l = 0
            pad_t = torch.floor(diff / 2)
        bboxes[:, 0] -= pad_l
        bboxes[:, 1] -= pad_t
        return bboxes


augmentation = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomResize(img_size=416),
    RandomSquarePad(),
    RandomColorJitter(brightness=0.4, saturation=0.4, hue=0.1),
    ToTensor(),
])


preprocess = Compose([
    Resize(img_size=416),
    SquarePad(),
    ToTensor(),
])
