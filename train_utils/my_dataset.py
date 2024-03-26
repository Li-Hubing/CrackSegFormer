import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(CrackDataset, self).__init__()
        flag = "train" if train else "val"

        data_root = os.path.join(root, flag)
        assert os.path.exists(data_root), "path '{}' does not exist.".format(data_root)

        imgs_root = os.path.join(data_root, "images")
        masks_root = os.path.join(data_root, "masks")

        self.images_list = os.listdir(imgs_root)
        self.images_path = [os.path.join(imgs_root, i) for i in self.images_list]
        # self.masks_path = [os.path.join(masks_root, i.split(".")[0] + ".png")
        #                    for i in self.images_list]  # mask.png  image_name/=mask_name
        self.masks_path = [os.path.join(masks_root, i) for i in self.images_list]  # same_name

        assert (len(self.images_path) == len(self.masks_path))

        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert('RGB')
        mask = Image.open(self.masks_path[idx]).convert('L')
        mask = np.array(mask) / 255

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.images_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中,channel,h,w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
