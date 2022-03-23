import os
import argparse
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
from tqdm import trange
from tensorboardX import SummaryWriter
from pycocotools.coco import COCO

from yolov3.dataset import DetectionDataset, DistributedMultiScaleSampler
from yolov3.transforms import augmentation, preprocess
from yolov3.models import YOLOs
from yolov3.models.utils import parse_weights
from yolov3.utils import evaluate


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logdir', default='./logs/yolov3',
                    help='log directory')
# dataset
parser.add_argument('--train_ann_file', type=str,
                    default='./data/coco/annotations/instances_trainvalno5k.json',
                    help='path to training annotation file')
parser.add_argument('--train_img_root', type=str,
                    default='./data/coco/all2014',
                    help='path to root of training images')
parser.add_argument('--val_ann_file', type=str,
                    default='./data/coco/annotations/instances_5k.json',
                    help='path to val annotation file')
parser.add_argument('--val_img_root', type=str,
                    default='./data/coco/all2014',
                    help='path to root of val images')
# learning
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--burn_in', default=1000, type=int,
                    help='warmup iters')
parser.add_argument('--steps', default=[400000, 450000], type=int, nargs='*',
                    help='lr decay iters')
parser.add_argument('--max_iters', default=500000, type=int,
                    help='number of iterations')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum')
parser.add_argument('--decay', default=0.0005, type=float,
                    help='L2 regularization for convolution kernel')
parser.add_argument('--batch_size', default=4, type=int,
                    help='batch size')
parser.add_argument('--accumulation', default=16, type=int,
                    help='batch accumulation')
parser.add_argument('--num_workers', default=6, type=int,
                    help='dataloader workers')
parser.add_argument('--scale_range', default=[320, 608], type=int, nargs=2,
                    help='rnage of image size for multi-scale training')
parser.add_argument('--scale_interval', default=10, type=int)
parser.add_argument('--ckpt_interval', default=2000, type=int)
parser.add_argument('--eval_interval', default=4000, type=int)
# yolo
parser.add_argument('--model', default='yolov3', choices=YOLOs.keys(),
                    help='model name')
parser.add_argument('--weights', default=None, type=str,
                    help='path to pretrained weights')
parser.add_argument('--n_classes', default=80, type=int,
                    help='nunmber of classes')
parser.add_argument('--ignore_threshold', default=0.7, type=float,
                    help='ignore iou threshold')
parser.add_argument('--conf_threshold', default=0.005, type=float,
                    help='evaluation conf threshold')
parser.add_argument('--nms_threshold', default=0.45, type=float,
                    help='evaluation nms threshold')
parser.add_argument('--img_size', default=416, type=int,
                    help='evaluation image size')
parser.add_argument('--val_batch_size', default=32, type=int,
                    help='evaluation batch size')
args = parser.parse_args()


def inflooper(dataloader, sampler):
    epoch = 0
    while True:
        # shuffle and randomly generate image sizes at the stat of each epoch
        sampler.set_epoch(epoch)
        for data in dataloader:
            yield data
        epoch += 1


def main(rank, world_size):
    device = torch.device('cuda:%d' % rank)
    if rank == 0:
        os.makedirs(args.logdir)
        # tensorboard writer
        writer = SummaryWriter(args.logdir)

    # train dataset
    train_dataset = DetectionDataset(
        COCO(args.train_ann_file),
        args.train_img_root,
        img_size=None,              # changed by DistributedMultiScaleSampler
        transforms=augmentation
    )
    multiscale_sampler = DistributedMultiScaleSampler(
        train_dataset, args.batch_size * args.accumulation,
        args.scale_interval, args.scale_range)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,
        sampler=multiscale_sampler,
        num_workers=args.num_workers,
        collate_fn=DetectionDataset.collate)

    # test dataset
    test_dataset = DetectionDataset(
        COCO(args.val_ann_file),
        args.val_img_root,
        args.img_size,
        transforms=preprocess)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        collate_fn=DetectionDataset.collate)

    # model
    model = YOLOs[args.model](args.n_classes, args.ignore_threshold).to(device)
    # optimizer
    decay_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'conv.weight' in name:
            decay_params.append(param)
        else:
            other_params.append(param)
    optim = torch.optim.SGD([
        {"params": decay_params, "weight_decay": args.decay},
        {"params": other_params, "weight_decay": 0},
    ], lr=args.lr, momentum=args.momentum)

    def lr_factor(iter_i):
        if iter_i < args.burn_in:
            return pow(iter_i / args.burn_in, 4)
        factor = 1.0
        for step in args.steps:
            if iter_i >= step:
                factor *= 0.1
        return factor
    # learning rate scheduler
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_factor)

    iter_state = 1
    # load weights
    if rank == 0 and args.weights:
        if args.weights.endswith('.pt'):
            print("loading pytorch weights:", args.weights)
            ckpt = torch.load(args.weights)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            sched.load_state_dict(ckpt['sched'])
            iter_state = ckpt['iter'] + 1
        else:
            print("loading darknet weights:", args.weights)
            parse_weights(model, args.weights)
    dist.barrier()          # it takes time to load weights
    model = DDP(model, device_ids=[rank], output_device=rank)

    # start training loop
    looper = inflooper(train_loader, multiscale_sampler)
    if rank == 0:
        pbar = trange(iter_state, args.max_iters + 1, ncols=0)
    else:
        pbar = range(iter_state, args.max_iters + 1)
    for iter_i in pbar:
        # accumulation loop
        optim.zero_grad()
        loss_record = defaultdict(float)
        for _ in range(args.accumulation):
            imgs, targets, _, _ = next(looper)
            imgs, targets = imgs.to(device), targets.to(device)
            loss_dict = model(imgs, targets)
            loss = 0
            for name, value in loss_dict.items():
                loss_record[name] += value.detach().item()
                loss += value
            # normalize the loss by the effective batch size
            loss = loss / (imgs.shape[0] * args.accumulation)
            loss.backward()
        optim.step()
        sched.step()

        # logging
        if rank == 0:
            for key, value in loss_record.items():
                writer.add_scalar('train/%s' % key, value, iter_i)
            writer.add_scalar('train/lr', sched.get_last_lr()[0], iter_i)
            # update progress bar
            postfix_str = ", ".join(
                '%s: %.3f' % (k, v) for k, v in loss_record.items())
            pbar.set_postfix_str(postfix_str)

        # evaluation
        if iter_i % args.eval_interval == 0:
            if rank == 0:
                model.eval()
                ap50_95, ap50 = evaluate(
                    model=model.module,
                    loader=test_loader,
                    conf_threshold=args.conf_threshold,
                    nms_threshold=args.nms_threshold,
                    device=device)
                model.train()
                writer.add_scalar('val/COCO_AP50', ap50, iter_i)
                writer.add_scalar('val/COCO_AP50_95', ap50_95, iter_i)
            dist.barrier()

        # save checkpoint
        if iter_i % args.ckpt_interval == 0:
            if rank == 0:
                ckpt = {
                    'iter': iter_i,
                    'model': model.module.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                }
                torch.save(
                    ckpt, os.path.join(args.logdir, "ckpt%d.pt" % iter_i))
            dist.barrier()
    if rank == 0:
        pbar.close()
        writer.close()


def initialize_process(rank, world_size):
    import datetime
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "39846"

    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=30),
                            world_size=world_size, rank=rank)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    print("Node %d is initialized" % rank)
    main(rank, world_size)


def spawn_process():
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) is not None:
        world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        world_size = 1

    processes = []
    for rank in range(world_size):
        p = Process(target=initialize_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    spawn_process()
