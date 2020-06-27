import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')

class INAT(data.Dataset):
    def __init__(self, root, ann_file, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [224, 224]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.resize_aug = transforms.Resize(256)
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        img = self.loader(path)
        species_id = self.classes[index]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.resize_aug(img)
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)

        return img, species_id

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    data_dir = "/home/ubuntu/inat/"

    train_dataset = INAT(root=data_dir, ann_file=data_dir+"new_train.json", is_train=True)
    val_dataset = INAT(root=data_dir, ann_file=data_dir+"new_val.json", is_train=True)
    test_dataset = INAT(root=data_dir, ann_file=data_dir+"new_test.json", is_train=False)

    print(len(train_dataset), train_dataset[0])
    print(len(val_dataset), val_dataset[0])
    print(len(test_dataset), test_dataset[0])
