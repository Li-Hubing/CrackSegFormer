import os
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer
from models.unet.unet import UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.unet.vgg_unet import VGG16UNet
from models.fcn.fcn import fcn_resnet101


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(512),
         transforms.ToTensor(),
         transforms.Normalize([0.473, 0.493, 0.504], [0.100, 0.100, 0.099])])

    # load image
    imgs_root = "dataset/test/val/images"
    images_list = os.listdir(imgs_root)
    prediction_save_path = "dataset/test/val/pred"

    # create model
    model = SegFormer(num_classes=num_classes, phi="b0")
    # model = VGG16UNet(num_classes=num_classes)
    # model = MobileV3Unet(num_classes=num_classes)

    # load model weights
    weights_path = "logs/20221028-073013-best_model.pth"
    # weights_path = "logs/VGG16UNet.pth"
    # weights_path = "logs/MobileV3Unet.pth"
    # weights_path = "logs/fcn_resnet101.pth"
    # weights_path = "logs/deeplabv3_resnet101.pth"
    # weights_path = "logs/deeplabv3_mobilenetv3_large.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."

    pretrain_weights = torch.load(weights_path, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    # model = fcn_resnet101(aux=False, num_classes=num_classes)
    # model = deeplabv3_resnet101(aux=False, num_classes=num_classes)
    # model = deeplabv3_mobilenetv3_large(aux=False, num_classes=num_classes)
    # # deeplabv3:delete weights about aux_classifier
    # # weights_dict = torch.load(args.weights, map_location='cpu')['model']
    # weights_dict = torch.load(weights_path, map_location='cpu')
    # for k in list(weights_dict.keys()):
    #     if "aux" in k:
    #         del weights_dict[k]
    #
    # model.load_state_dict(weights_dict)
    # model.to(device)

    # prediction
    model.eval()
    with torch.no_grad():
        for index, image in enumerate(images_list):
            original_img = Image.open(os.path.join(imgs_root, image)).convert("RGB")
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            output = model(img.to(device))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            prediction[prediction == 1] = 255
            mask = Image.fromarray(prediction)
            mask.save(os.path.join(prediction_save_path, image.split('.')[0] + '.png'))
            print("\r[{}] processing [{}/{}]".format(image, index + 1, len(images_list)), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
