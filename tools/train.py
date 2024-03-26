import os
import time
import datetime
import torch
import numpy as np

from torch.utils.data import DataLoader
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from train_utils.my_dataset import CrackDataset
import train_utils.transforms as T
from train_utils.utils import plot, show_config

from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.fcn.fcn import fcn_resnet50
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # self.transforms = T.Compose([
        #     T.Resize(base_size),
        #     T.RandomHorizontalFlip(hflip_prob),
        #     T.RandomCrop(img_size),
        #     T.ToTensor(),
        #     T.Normalize(mean=mean, std=std)
        # ])
        min_size = int(0.8 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, img_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    img_size = 512

    if train:
        return SegmentationPresetTrain(base_size, img_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(img_size, mean=mean, std=std)


def create_model(aux, num_classes, pretrained=True):
    # model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    # model = fcn_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    # model = deeplabv3_resnet101(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)
    model = deeplabv3_mobilenetv3_large(aux=aux, num_classes=num_classes, pretrain_backbone=pretrained)

    if args.pretrained_weights != "":
        weights_dict = torch.load(args.pretrained_weights, map_location='cpu')
        for k in list(weights_dict.keys()):
            if "classifier.4" in k:
                del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    mean = (0.473, 0.493, 0.504)
    std = (0.100, 0.100, 0.099)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    train_dataset = CrackDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = CrackDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,  # 不为1时报错,可能是transform的设置问题
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)

    # model = SegFormer(num_classes=num_classes, phi=args.phi, pretrained=args.pretrained)
    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = MobileV3Unet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    # model = VGG16UNet(num_classes=num_classes, pretrain_backbone=args.pretrained)
    model = create_model(aux=args.aux, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    if args.pretrained_weights != "":
        assert os.path.exists(args.pretrained_weights), "weights file: '{}' not exist.".format(args.pretrained_weights)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained_weights, map_location=device)["state_dict"]
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        print("load_key: ", load_key)
        print("no_load_key: ", no_load_key)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    # params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # fcn,deeplabv3优化参数
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]
    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = {
        'adam': torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                 weight_decay=args.weight_decay),
        'adamw': torch.optim.AdamW(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999),
                                   weight_decay=args.weight_decay),
        'sgd': torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    }[args.optimizer_type]

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=20)

    # 查看学习率
    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 用来保存训练以及验证过程中信息
    record_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = "../logs/{}-results.txt".format(record_time)

    config_info = {
        'record_time': record_time,
        'device': args.device,
        'data_path': args.data_path,
        'num_classes': num_classes,
        'model': model.__class__.__name__,
        'backbone_pretrained': args.pretrained,
        'pretrained_weights': args.pretrained_weights,
        "loss": "cross_entropy(weight=[1.0,2.0])+dice_loss",
        'optimizer_type': args.optimizer_type,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'img_size': '512 * 512',
        'start_epoch': args.start_epoch,
        'epochs': args.epochs,
        "warmup_epochs: 20\n"
        'weights_save_best': args.save_best,
        'amp': args.amp,
        'num_workers': num_workers
    }

    show_config(config_info)

    with open(results_file, "a") as f:
        f.write("Configurations:\n")
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n\n")

    # 训练过程可视化
    train_loss = []
    dice_coefficient = []
    img_save_path = "../logs/{}-visualization.svg".format(record_time)

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.8f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        # 绘图
        train_loss.append(mean_loss)
        dice_coefficient.append(dice)
        plot(train_loss, dice_coefficient, img_save_path)

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        if args.save_best is True:
            torch.save(model.state_dict(), "../logs/{}-best_model.pth".format(record_time))
            best_model_info = "../logs/{}-best_model_info.txt".format(record_time)
            with open(best_model_info, "w") as f:
                f.write(train_info + val_info)
        else:
            torch.save(model.state_dict(), "../logs/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--device", default="cuda:0", help="training device")

    parser.add_argument("--data-path",
                        default="../dataset",
                        help="root")
    parser.add_argument("--num-classes", default=1, type=int)  # exclude background

    parser.add_argument("--aux", default=True, type=bool, help="deeplabv3 auxilier loss")
    parser.add_argument("--phi", default="b0", help="Use backbone")
    parser.add_argument('--pretrained', default=True, type=bool, help='backbone')
    parser.add_argument('--pretrained-weights', type=str,
                        default="",
                        help='pretrained weights path')

    parser.add_argument('--optimizer-type', default="adamw")
    parser.add_argument('--lr', default=0.00006, type=float, help='initial learning rate')
    parser.add_argument('--warmup-epochs', default=20, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for automatic mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
