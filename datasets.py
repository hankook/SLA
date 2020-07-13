import os, csv, random, torch, numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import BatchSampler as BaseSampler
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, CelebA

class BatchSampler(BaseSampler):
    def __init__(self, dataset, num_iterations, batch_size):

        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        self.sampler = None

    def __iter__(self):
        indices = []
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)),
                                    self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations

_transform_large_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
_transform_large_test  = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])

def check_dataset(name, root, split, num_samples_per_class=None):
    if name == 'inat':
        import inat
        if not root.endswith('/'):
            root = root + '/'
        if split == 'train':
            dataset = inat.INAT(root, os.path.join(root, 'train2018.json'), is_train=True)
        else:
            dataset = inat.INAT(root, os.path.join(root, 'val2018.json'), is_train=False)
        return dataset

    elif name == 'imagenet':
        if split == 'train':
            dataset = ImageFolder(os.path.join(root, 'train'), _transform_large_train)
            if num_samples_per_class is not None:
                indices = np.load('splits/imagenet_{}.npy'.format(num_samples_per_class))
                dataset = Subset(dataset, indices)
            return dataset
        else:
            return ImageFolder(os.path.join(root, 'val'), _transform_large_test)

    elif name.startswith('imagenet'):
        n = int(name[len('imagenet'):])
        labels = torch.load('splits/imagenet_labels_{}.pth'.format(n))
        def target_transform(y):
            return labels.index(y)

        if split == 'train':
            dataset = ImageFolder(os.path.join(root, 'train'), _transform_large_train, target_transform=target_transform)
            if num_samples_per_class is not None:
                indices = np.load('splits/imagenet_{}.npy'.format(num_samples_per_class))
            else:
                indices = list(range(len(dataset)))
            indices = [i for i in indices if dataset.samples[i][1] in labels]
            dataset = Subset(dataset, indices)
        else:
            dataset = ImageFolder(os.path.join(root, 'val'), _transform_large_test, target_transform=target_transform)
            indices = list(range(len(dataset)))
            indices = [i for i in indices if dataset.samples[i][1] in labels]
            dataset = Subset(dataset, indices)

        return dataset

    elif name == 'tiny-imagenet':
        if split == 'train':
            return ImageFolder(os.path.join(root, 'train'), _transform_large_train)
        else:
            return ImageFolder(os.path.join(root, 'val2'),  _transform_large_test)

    elif name.startswith('cub200') or name.startswith('indoor') or name.startswith('dogs'):
        if split == 'train':
            return ImageFolder(os.path.join(root, 'train'), _transform_large_train)
        else:
            return ImageFolder(os.path.join(root, 'test'), _transform_large_test)

    elif name.startswith('cifar'):
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])
        test_transform  = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                   (0.5, 0.5, 0.5))])

        CIFAR = CIFAR10 if name == 'cifar10' else CIFAR100
        if split == 'train':
            if num_samples_per_class is None:
                indices = [i for i in range(50000) if i not in np.load('splits/{}_val_idx.npy'.format(name))]
            else:
                indices = np.load('splits/{}_{}_train_idx.npy'.format(name,
                                                                      num_samples_per_class))
            dataset = Subset(CIFAR(root, train=True, download=True, transform=train_transform), indices)
        elif split == 'val':
            indices = np.load('splits/{}_val_idx.npy'.format(name))
            dataset = Subset(CIFAR(root, train=True, download=True, transform=test_transform), indices)
        else:
            dataset = CIFAR(root, train=False, download=True, transform=test_transform)

        return dataset

    else:
        raise Exception('Unknown dataset {}'.format(name))

def check_dataloader(dataset, num_iterations, *args):
    sampler = BatchSampler(dataset, num_iterations, *args)
    return DataLoader(dataset, num_workers=12, batch_sampler=sampler)
