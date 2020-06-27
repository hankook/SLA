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
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--datadir', type=str, default='/home/ubuntu/ILSVRC/Data/CLS-LOC')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--num-iterations', type=int, default=900000)
    parser.add_argument('--num-samples-per-class', type=int, default=None)
    parser.add_argument('--val-freq', type=int, default=10000)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--aug', type=str, default=None)

    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--with-training-agg', action='store_true')
    parser.add_argument('--with-large-loss', action='store_true')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--no-wd-bn', action='store_true')
    parser.add_argument('--no-replace', action='store_true')

    args = parser.parse_args()

    suffix = "" if args.logdir is None else "_" + args.logdir
    args.logdir = os.path.join('logs', args.dataset, args.model,
                               '{}_{}{}'.format(args.mode, args.aug, suffix))
    utils.set_logging_defaults(args)
    logger = logging.getLogger('main')
    writer = SummaryWriter(args.logdir)

    ### DataLoader
    trainloader = check_dataloader(check_dataset(args.dataset, args.datadir, 'train',
                                                 num_samples_per_class=args.num_samples_per_class),
                                   args.val_freq, args.batchsize, not args.no_replace)
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
    elif args.mode in ['sla', 'sla+random']:
        model = check_model(args.model, n*m).to(device)
    elif args.mode == 'sla+sd':
        model = check_model(args.model, n*m, n).to(device)
    else:
        raise Exception('unknown mode: {}'.format(args.mode))

    if args.sync_bn:
        # torch.distributed.init_process_group('nccl')
        torch.distributed.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
        # process_group = torch.distributed.new_group()
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.DataParallel(model)
    model.num_classes = n
    if args.aug is not None:
        model.num_transformations = m
    if not args.no_wd_bn:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        param_groups = [ { 'params': [], 'weight_decay': args.wd },
                         { 'params': [], 'weight_decay': 0 } ]
        for name, p in model.named_parameters():
            if 'bn' in name:
                param_groups[1]['params'].append(p)
            else:
                param_groups[0]['params'].append(p)
        optimizer = optim.SGD(param_groups, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    if not args.dataset.startswith('imagenet') and args.dataset != 'inat':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      [args.num_iterations // 2,
                                                       args.num_iterations*3 // 4])
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 args.num_iterations // 3,
                                                 0.1)

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
    elif args.mode == 'sla+random':
        builder = partial(trainers.create_sla_random_trainer, model, transform,
                          with_large_loss=args.with_large_loss,
                          device=device)
    else:
        raise NotImplementedError('not implemented mode: {}'.format(args.mode))

    train    = builder(optimizer=optimizer, name='train')
    validate = builder(optimizer=None,      name='val')
    test     = builder(optimizer=None,      name='test')


    @train.on(Events.ITERATION_COMPLETED)
    def adjust_learning_rate(engine):
        lr_scheduler.step()

    for engine in [train, validate, test]:
        engine.add_event_handler(Events.ITERATION_COMPLETED, utils.log_loss)
        engine.add_event_handler(Events.EPOCH_COMPLETED, utils.log_metrics, writer, train)

    train.best = None

    @train.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        validate.run(valloader, 1)
        test.run(testloader, 1)

        key = 'single_acc'
        if engine.best is None or engine.best[0] < validate.state.metrics[key]:
            engine.best = (validate.state.metrics[key], test.state.metrics[key])
            logger.info('[iteration {}] [BEST] [val {:.4f}] [test {:.4f}]'.format(
                engine.state.iteration, *engine.best))
            torch.save({
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'best': train.best,
                'args': args,
                'epoch': engine.state.epoch,
                'iteration': engine.state.iteration,
            }, os.path.join(args.logdir, 'model-best.pth'))

    ### Resume
    if args.resume is not None:
        ckpt = torch.load(os.path.join(args.resume, 'model-best.pth'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        for pg in optimizer.param_groups:
            pg['weight_decay'] = args.wd
        train.best = ckpt['best']

        @train.on(Events.STARTED)
        def init_engine(engine):
            engine.state.epoch = ckpt['epoch']
            engine.state.iteration = ckpt['iteration']

        for _ in range(ckpt['iteration']):
            lr_scheduler.step()

    ### Training
    train.run(trainloader, args.num_iterations // args.val_freq)

if __name__ == '__main__':
    main()
