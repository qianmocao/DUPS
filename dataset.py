import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#from transformers import AutoImageProcessor
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, WeightedRandomSampler

class RVL(Dataset):
    def __init__(self, img_dir, label_dir, split,data_aug=False): # split: train, valid, test
        self.split = split
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.data_aug = data_aug

        data_all = np.load(label_dir, allow_pickle=True).item()
        self.data = data_all[split]

        #self.img_proc = AutoImageProcessor.from_pretrained('microsoft/resnet-18')
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.img_proc_aug = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(0.75, 1.25), antialias=True),
            #v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=5, shear=(-10, 10, -10, 10)),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 5)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_url = record[0]
        clas = record[1]
        bbox = record[2]
        gt = int(clas)
        img = Image.open(f'{self.img_dir}{img_url}').convert('RGB')

        if self.split == 'train' and self.data_aug:
            img_feat = self.img_proc_aug(img)
        else:
            img_feat = self.img_proc(img)

        sample_info = {'img_url': img_url,
                       'img': img_feat,
                       'label': gt,
                       'bbox': bbox,
                       }
        return sample_info


def loadData(data_path, data_aug=False):
    data_dir = dict()
    img_dir = os.path.join(data_path, 'images/')
    label_dir = os.path.join(data_path, 'RVL_CDIP_resnet.npy')
    for split in ['train', 'valid', 'test']:
        data_dir[split] = RVL(img_dir, label_dir, split,data_aug)
    return data_dir
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    res = loadData()
    for i in range(10):
        data = next(iter(res['train']))
        print(f"{data['img_url']}, {data['label']}")
        img = torch.permute(data['img'], (1,2,0))
        ret = img.numpy()
        ret = (ret - ret.min())/(ret.max()-ret.min())
        fig, ax = plt.subplots()
        im = ax.imshow(ret)
        plt.show()
