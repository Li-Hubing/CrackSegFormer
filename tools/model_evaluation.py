import torch
import time
import datetime
from thop import profile
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

from torchvision.models import resnet18
from models.fcn.fcn import fcn_resnet50
from models.fcn.fcn import fcn_resnet101
from models.unet.unet import UNet
from models.unet.vgg_unet import VGG16UNet
from models.unet.mobilenet_unet import MobileV3Unet
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet50
from models.deeplab_v3.deeplabv3 import deeplabv3_resnet101
from models.deeplab_v3.deeplabv3 import deeplabv3_mobilenetv3_large
from models.segformer.segformer import SegFormer


num_classes = 2
input = torch.randn(1, 3, 512, 512)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = [fcn_resnet50(aux=False, num_classes=num_classes),
          fcn_resnet101(aux=False, num_classes=num_classes),
          UNet(in_channels=3, num_classes=num_classes, base_c=64),
          VGG16UNet(num_classes=num_classes),
          MobileV3Unet(num_classes=num_classes),
          deeplabv3_resnet50(aux=False, num_classes=num_classes),
          deeplabv3_resnet101(aux=False, num_classes=num_classes),
          deeplabv3_mobilenetv3_large(aux=False, num_classes=num_classes),
          SegFormer(num_classes=num_classes, phi="b0"),
          SegFormer(num_classes=num_classes, phi="b1"),
          SegFormer(num_classes=num_classes, phi="b2"),
          SegFormer(num_classes=num_classes, phi="b3"),
          SegFormer(num_classes=num_classes, phi="b4"),
          SegFormer(num_classes=num_classes, phi="b5")]
model_list = ["fcn_resnet50", "fcn_resnet101", "UNet", "VGG16UNet", "MobileV3Unet", "deeplabv3_resnet50",
              "deeplabv3_resnet101", "deeplabv3_mobilenetv3_large", "SegFormer B0", "SegFormer B1", "SegFormer B2",
              "SegFormer B3", "SegFormer B4", "SegFormer B5"]

# record_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# results_file = "../logs/{}-results.txt".format(record_time)

for i in range(14):
    model = models[i]
    flops, params = profile(model, (input,))

    model.to(device)
    model.eval()  # 关掉dropout方法
    total_time = 0.
    Latency = 0.
    n = 100
    with torch.no_grad():
        init_img = torch.zeros((1, 3, 512, 512), device=device)
        model(init_img)
        for m in range(n):
            t_start = time.time()
            output = model(input.to(device))
            t_end = time.time()
            total_time += t_end - t_start
            
    Latency = total_time / n
    FPS = 1 / Latency
    print('{}'.format(model_list[i]))
    print(f"Params:{params / 1e6:.1f}M")
    print(f"FLOPs:{flops / 1e9:.1f}G")
    print(f"Latency:{Latency * 1e3:.1f}")
    print(f"FPS: {FPS:.1f}")
    print('-' * 20)
    # with open(results_file, "a") as f:
    #     # 记录每个epoch对应的train_loss、lr以及验证集各指标
    #     info = f"[model:{model_list[i]}]\n" \
    #            f"Params:{params / 1e6:.3f}M\n" \
    #            f"FLOPs:{flops / 1e9:.3f}G\n" \
    #            f"Latency:{Latency * 1e3:.3f}\n" \
    #            f"FPS: {FPS:.3f}\n"
    #     f.write(info + "\n\n")
