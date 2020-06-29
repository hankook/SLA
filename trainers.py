import torch, copy, random, numpy as np
from torch.nn import functional as F
from torch import nn
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from ignite import metrics

import utils

device = torch.device('cuda:0')

def create_baseline_trainer(model,
                            optimizer=None,
                            name='train',
                            device=None):

    if device is not None:
        model.to(device)

    is_train = optimizer is not None
    def _update(engine, batch):
        model.train(is_train)

        with torch.set_grad_enabled(is_train):
            images, labels = convert_tensor(batch, device=device)
            preds = model(images)
            loss = F.cross_entropy(preds, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {
            'loss': loss.item(),
            'y_pred': preds,
            'y': labels
        }

    engine = Engine(_update)
    engine.name = name
    metrics.Average(lambda o: o['loss']).attach(engine, 'single_loss')
    metrics.Accuracy(lambda o: (o['y_pred'], o['y'])).attach(engine, 'single_acc')
    return engine


def create_sla_trainer(model,
                       transform,
                       optimizer=None,
                       with_large_loss=False,
                       name='train',
                       device=None):

    if device is not None:
        model.to(device)

    is_train = optimizer is not None
    def _update(engine, batch):
        model.train(is_train)

        with torch.set_grad_enabled(is_train):
            images, labels = convert_tensor(batch, device=device)
            batch_size = images.shape[0]
            images = transform(model, images, labels)
            n = images.shape[0] // batch_size

            preds = model(images)
            labels = torch.stack([labels*n+i for i in range(n)], 1).view(-1)
            loss = F.cross_entropy(preds, labels)
            if with_large_loss:
                loss = loss * n

            single_preds = preds[::n, ::n]
            single_labels = labels[::n] // n

            agg_preds = 0
            for i in range(n):
                agg_preds = agg_preds + preds[i::n, i::n] / n

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {
            'loss': loss.item(),
            'preds': preds,
            'labels': labels,
            'single_preds': single_preds,
            'single_labels': single_labels,
            'agg_preds': agg_preds,
        }

    engine = Engine(_update)
    engine.name = name

    metrics.Average(lambda o: o['loss']).attach(engine, 'total_loss')
    metrics.Accuracy(lambda o: (o['preds'], o['labels'])).attach(engine, 'total_acc')

    metrics.Average(lambda o: F.cross_entropy(o['single_preds'], o['single_labels'])).attach(engine, 'single_loss')
    metrics.Accuracy(lambda o: (o['single_preds'], o['single_labels'])).attach(engine, 'single_acc')

    metrics.Average(lambda o: F.cross_entropy(o['agg_preds'], o['single_labels'])).attach(engine, 'agg_loss')
    metrics.Accuracy(lambda o: (o['agg_preds'], o['single_labels'])).attach(engine, 'agg_acc')

    return engine


def create_sla_sd_trainer(model,
                          transform,
                          optimizer=None,
                          T=1.0,
                          with_large_loss=False,
                          name='train',
                          device=None):

    if device is not None:
        model.to(device)

    is_train = optimizer is not None
    def _update(engine, batch):
        model.train(is_train)

        with torch.set_grad_enabled(is_train):
            images, single_labels = convert_tensor(batch, device=device)
            batch_size = images.shape[0]
            images = transform(model, images, single_labels)
            n = images.shape[0] // batch_size

            joint_preds, single_preds = model(images, None)
            single_preds = single_preds[::n]
            joint_labels = torch.stack([single_labels*n+i for i in range(n)], 1).view(-1)

            joint_loss = F.cross_entropy(joint_preds, joint_labels)
            single_loss = F.cross_entropy(single_preds, single_labels)
            if with_large_loss:
                joint_loss = joint_loss * n

            agg_preds = 0
            for i in range(n):
                agg_preds = agg_preds + joint_preds[i::n, i::n] / n

            distillation_loss = F.kl_div(F.log_softmax(single_preds / T, 1),
                                         F.softmax(agg_preds.detach() / T, 1),
                                         reduction='batchmean')

            loss = joint_loss + single_loss + distillation_loss.mul(T**2)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {
            'loss': loss.item(),
            'preds': joint_preds,
            'labels': joint_labels,
            'single_preds': single_preds,
            'single_labels': single_labels,
            'agg_preds': agg_preds,
        }

    engine = Engine(_update)
    engine.name = name

    metrics.Average(lambda o: o['loss']).attach(engine, 'total_loss')
    metrics.Accuracy(lambda o: (o['preds'], o['labels'])).attach(engine, 'total_acc')

    metrics.Average(lambda o: F.cross_entropy(o['single_preds'], o['single_labels'])).attach(engine, 'single_loss')
    metrics.Accuracy(lambda o: (o['single_preds'], o['single_labels'])).attach(engine, 'single_acc')

    metrics.Average(lambda o: F.cross_entropy(o['agg_preds'], o['single_labels'])).attach(engine, 'agg_loss')
    metrics.Accuracy(lambda o: (o['agg_preds'], o['single_labels'])).attach(engine, 'agg_acc')

    return engine

