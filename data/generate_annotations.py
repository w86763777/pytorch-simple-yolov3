import json
import os

from pycocotools.coco import COCO
from tqdm import tqdm


def darknet2coco(coco_train, coco_val, darknet_split_path):
    train_ids = set(coco_train.getImgIds())
    val_ids = set(coco_val.getImgIds())
    coco = {
        "images": [],
        "annotations": [],
        "categories": coco_train.dataset["categories"],
        "info": coco_train.dataset["info"],
        "licenses": coco_train.dataset["licenses"],

    }
    with open(darknet_split_path, 'r') as f:
        for line in tqdm(f.readlines(), desc=darknet_split_path):
            name = os.path.splitext(os.path.basename(line))[0]
            a, b, id = name.split('_')
            id = int(id)
            if id in train_ids:
                anno_ids = coco_train.getAnnIds(imgIds=id)
                annos = coco_train.loadAnns(anno_ids)
                coco['annotations'].extend(annos)
                coco["images"].append(coco_train.loadImgs(id)[0])
            else:
                assert id in val_ids
                anno_ids = coco_val.getAnnIds(imgIds=id)
                annos = coco_val.loadAnns(anno_ids)
                coco['annotations'].extend(annos)
                coco["images"].append(coco_val.loadImgs(id)[0])
    coco['images'] = sorted(
        coco['images'], key=lambda x: x['id'])
    coco['annotations'] = sorted(
        coco['annotations'], key=lambda x: x['id'])

    return coco


if __name__ == '__main__':
    coco_train = COCO('./coco/annotations/instances_train2014.json')
    coco_val = COCO('./coco/annotations/instances_val2014.json')

    coco = darknet2coco(coco_train, coco_val, './coco/trainvalno5k.part')
    print("Saving to coco/annotations/instances_trainvalno5k.json")
    with open('./coco/annotations/instances_trainvalno5k.json', 'w') as f:
        json.dump(coco, f)

    coco = darknet2coco(coco_train, coco_val, './coco/5k.part')
    print("Saving to coco/annotations/instances_5k.json")
    with open('./coco/annotations/instances_5k.json', 'w') as f:
        json.dump(coco, f)
