import torch, torchvision
import os, argparse, logging, numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models import check_model
from datasets import check_dataset, check_dataloader
import trainers, utils
import augmentations
from functools import partial

from ignite.engine.engine import Engine, State, Events

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--datadir', type=str, default='/home/ubuntu/ILSVRC/Data/CLS-LOC')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--num-samples-per-class', type=int, default=None)

    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--aug', type=str, default=None)

    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--with-training-agg', action='store_true')
    parser.add_argument('--with-large-loss', action='store_true')

    args = parser.parse_args()

    ### DataLoader
    valloader   = DataLoader(check_dataset(args.dataset, args.datadir, 'train'),
                             batch_size=args.batchsize, shuffle=False, num_workers=8)
    valloader   = DataLoader(check_dataset(args.dataset, args.datadir, 'val'),
                             batch_size=args.batchsize, shuffle=False, num_workers=8)
    testloader  = DataLoader(check_dataset(args.dataset, args.datadir, 'test'),
                             batch_size=args.batchsize, shuffle=False, num_workers=8)

    ### Model
    if args.dataset.startswith('cifar'):
        n = int(args.dataset[5:])
    elif args.dataset == 'imagenet':
        n = 1000
    elif args.dataset.startswith('imagenet'):
        n = int(args.dataset[len('imagenet'):])
    elif args.dataset == 'cub200' or args.dataset == 'tiny-imagenet':
        n = 200
    elif args.dataset == 'indoor':
        n = 67
    elif args.dataset == 'dogs':
        n = 120
    elif args.dataset == 'inat':
        n = 8142

    ### Transformation
    if args.aug is not None:
        transform, m = augmentations.__dict__[args.aug]()

    ### Model
    if args.mode in ['baseline', 'da']:
        model = check_model(args.model, n).to(device)
    elif args.mode == 'mt':
        model = check_model(args.model, n, m).to(device)
    elif args.mode == 'sla':
        model = check_model(args.model, n*m).to(device)
    elif args.mode == 'sla+sd':
        model = check_model(args.model, n*m, n).to(device)
    else:
        raise Exception('unknown mode: {}'.format(args.mode))

    model = nn.DataParallel(model)

    ckpt = torch.load(args.ckpt)
    print(ckpt['iteration'])
    model.load_state_dict(ckpt['model'])

    ### Trainer
    if args.mode == 'baseline':
        builder = partial(trainers.create_baseline_trainer, model, device=device)
    elif args.mode == 'sla':
        builder = partial(trainers.create_sla_trainer, model, transform,
                          with_large_loss=args.with_large_loss,
                          device=device)
    elif args.mode == 'sla+sd':
        builder = partial(trainers.create_sla_sd_trainer, model, transform,
                          T=args.T,
                          with_training_agg=args.with_training_agg,
                          with_large_loss=args.with_large_loss,
                          device=device)
    else:
        raise NotImplementedError('not implemented mode: {}'.format(args.mode))

    validate = builder(optimizer=None,      name='val')
    test     = builder(optimizer=None,      name='test')

    test.run(testloader, 1)
    print(test.state.metrics)

if __name__ == '__main__':
    main()

