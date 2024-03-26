import os
import torch
from torch.utils import data
import time
import datetime
import numpy as np
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

from train_utils.my_dataset import CrackDataset
import train_utils.transforms as T
import torch.nn.functional as F
from train_utils.dice_coefficient_loss import multiclass_dice_coeff, build_target
import train_utils.distributed_utils as utils
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.fcn.fcn import fcn_resnet101


class SegmentationPresetEval:
    def __init__(self, mean=(0.473, 0.493, 0.504), std=(0.100, 0.100, 0.099)):
        self.transforms = T.Compose([
            T.Resize(512),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dices = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_time = 0.
    with torch.no_grad():
        init_img = torch.zeros((1, 3, 512, 512), device=device)
        model(init_img)

        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            t_start = time_synchronized()
            output = model(image)
            t_end = time_synchronized()
            total_time += t_end - t_start

            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

            pred = F.one_hot(output.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
            dice_target = build_target(target, num_classes, 255)
            dice = multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], 255)
            dices.append(dice.item())

        confmat.reduce_from_all_processes()

    return confmat, dices, total_time


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    num_classes = args.num_classes + 1

    val_dataset = CrackDataset(args.data_path, train=False,
                               transforms=SegmentationPresetEval()
                               )

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,  # must be 1
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=val_dataset.collate_fn)

    # model = UNet(in_channels=3, num_classes=num_classes, base_c=64)
    # model = VGG16UNet(num_classes=num_classes)
    # model = MobileV3Unet(num_classes=num_classes)
    model = SegFormer(num_classes=num_classes, phi=args.phi)
    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    # model = fcn_resnet101(aux=False, num_classes=num_classes)
    # # model = deeplabv3_resnet101(aux=False, num_classes=num_classes)
    # # # model = deeplabv3_mobilenetv3_large(aux=False, num_classes=num_classes)
    # weights_dict = torch.load(args.weights, map_location='cpu')
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]

    # model.load_state_dict(weights_dict)
    # model.to(device)

    record_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = "logs/{}-validation.txt".format(record_time)

    confmat, dices, total_time = evaluate(model, val_loader, device=device, num_classes=num_classes)

    predict_time_per_img = total_time / len(val_dataset)
    predict_imgs_per_time = len(val_dataset) / total_time
    print("predict time: {}".format(total_time))
    print("predict time per img: {}".format(predict_time_per_img))
    print("predict imgs_return per second: {}".format(predict_imgs_per_time))

    val_info = str(confmat)
    print(val_info)
    print(f"mean dice coefficient: {np.mean(dices):.3f}")
    print(f"std dice coefficient: {np.std(dices):.3f}")
    print("dice coefficient length: {}".format(len(dices)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("--device", default="cuda:0", help="training device")

    parser.add_argument("--data-path",
                        default="dataset/test",
                        help="images root")
    # parser.add_argument("--data-path",
    #                     default="./",
    #                     help="images root")
    parser.add_argument("--num-classes", default=1, type=int)  # exclude background

    parser.add_argument("--phi", default="b0", help="Use backbone")
    parser.add_argument("--weights", default="logs/20221028-073013-best_model.pth")
    # parser.add_argument("--weights", default="logs/deeplabv3_resnet101.pth")
    # parser.add_argument("--weights", default="logs/deeplabv3_mobilenetv3_large.pth")
    # parser.add_argument("--weights", default="logs/VGG16UNet.pth")
    # parser.add_argument("--weights", default="logs/MobileV3Unet.pth")
    # parser.add_argument("--weights", default="logs/segformer_b5.pth")

    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
