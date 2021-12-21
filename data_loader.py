import zipfile
import os
import numpy as np
from PIL import Image
from io import BytesIO

import torch
from torch.functional import Tensor
import torch.nn.functional as F
#import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

class ZipImageNetFolder(Dataset):
    """A data loader for ImageNet in zip file, the zip file should contain:

        zip_file/class_x/xxx.JPEG
        zip_file/class_x/yyy.JPEG
        ...
        zip_file/class_y/xxx.JPEG
        zip_file/class_y/yyy.JPEG

    Args:
        root (string): path to the zip file
        split (string, optional): The dataset split ('train' or 'val')
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        label_noise (float, optional): 
    
    Attributes:
        name_list (list): List of the image files names.
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of data tuples that may vary
    """


    def __init__(self, root:str, transform:str=None, label_noise: float=0.0) -> None:
        self.root = os.path.expanduser(root)
        self.zip_file = zipfile.ZipFile(self.root, 'r')
        self.transform = transform
        assert 0.0 <= label_noise and label_noise <=1.0, "noise proportion param out of [0, 1]"
        self.label_noise = label_noise
        
        
        self.name_list = []
        self.classes = []
        self.class_to_idx = {}
        self.num_classes = 0
        self.samples = []
        self.to_tensor = ToTensor()
        
        self.parse_archives()
        self.make_dataset()
        
    def parse_archives(self):
        name_list = self.zip_file.namelist()
        
        for name in name_list:
            if len(name.split('/')[1]) == 0:
                self.classes.append(name.split('/')[0])
            else:
                self.name_list.append(name)
        
        self.name_list.sort()
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
    
    def make_dataset(self):
        for name in self.name_list:
            class_name = name.split('/')[0]
            class_index = self.class_to_idx[class_name]
            item = name, class_index
            self.samples.append(item)
    
    def __getitem__(self, index: int) -> Tensor:
        buf = self.zip_file.read(name=self.samples[index][0])
        fh = BytesIO(buf)
        img = Image.open(fh)
        img.convert('RGB')
        x = img
        if self.transform is not None:
            x = self.transform(x)
        # ! TODO: change back to
        #y = F.one_hot(torch.tensor(self.samples[index][1]), num_classes=self.num_classes)
        y = torch.tensor(self.samples[index][1])
        
        # ! TODO: noisy_y is also an one-hot
        #noisy_y = (1.0 - self.label_noise) * y + self.label_noise * F.softmax(torch.randn(self.num_classes), dim=0)
        
        return x, y
    
    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    zip_file = '/home/sg955/rds/hpc-work/ImageNet/train.zip'
    loader = ZipImageNetFolder(zip_file, label_noise=0.8)
    print(loader.classes)
    print(loader.class_to_idx)
    print(loader.name_list)
    print(loader.__getitem__(0))
    print(loader.__getitem__(1))
    print(loader.__getitem__(2))