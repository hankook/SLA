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
                          with_training_agg=False,
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
            if with_training_agg:
                loss = loss + F.cross_entropy(agg_preds, single_labels)

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


def create_sla_random_trainer(model,
                       transform,
                       optimizer=None,
                       with_large_loss=False,
                       name='train',
                       device=None):

    if device is not None:
        model.to(device)

    is_train = optimizer is not None
    indices = [4, 22, 10, 2]
    # indices = [21, 3, 9, 19, 15, 13, 1, 7, 18, 12, 6, 0, 23, 11, 17, 5, 14, 20, 8, 16, 4, 22, 10, 2]
    def _update(engine, batch):
        model.train(is_train)

        with torch.set_grad_enabled(is_train):
            images, labels = convert_tensor(batch, device=device)
            batch_size = images.shape[0]
            images = transform(model, images, labels)
            n = images.shape[0] // batch_size

            sample_indices = []
            sample_labels = []
            for i in range(batch_size):
                for j, k in enumerate(indices):
                    sample_indices.append(i*n+k)
                    sample_labels.append(labels[i]*4+j)
            images = images[sample_indices]
            labels = torch.stack(sample_labels)
            preds = model(images)
            preds = preds[:, :(preds.shape[1] // n * 4)]
            loss = F.cross_entropy(preds, labels)

            n = 4

            if with_large_loss:
                loss = loss * (4 if is_train else n)

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


def create_sla_hardest_trainer(model,
                       transform,
                       optimizer=None,
                       with_large_loss=False,
                       name='train',
                       device=None):

    if device is not None:
        model.to(device)

    is_train = optimizer is not None
    _data = {
    }
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

            with torch.no_grad():
                losses = F.cross_entropy(preds, labels, reduction='none')
                losses = losses.view(batch_size, n).mean(0)
                _, hard_indices = losses.sort(0, descending=True)
                hard_indices.detach()

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


def create_supervised_aug_trainer(model,
                                  transform,
                                  optimizer,
                                  joint=True,
                                  beta=1.0,
                                  name='train',
                                  device=None):

    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.train()

        images, labels = convert_tensor(batch, device=device)
        batch_size = images.shape[0]
        images = transform(model, images, labels)
        n = images.shape[0] // batch_size

        if joint:
            preds = model(images)
            labels = torch.cat([labels*n+i for i in range(n)], 0)
            loss = F.cross_entropy(preds, labels)

            single_preds = preds.view(n, batch_size, -1, n)[0, :, :, 0]
            single_labels = labels[:batch_size] // n

        else:
            aug_labels = []
            for i in range(n):
                aug_labels = aug_labels + [i]*batch_size
            aug_labels = convert_tensor(torch.tensor(aug_labels), device=device)

            preds, aug_preds = model(images, None)
            labels = torch.cat([labels for i in range(n)], 0)
            loss = F.cross_entropy(preds, labels) + F.cross_entropy(aug_preds, aug_labels).mul(beta)

            single_preds = preds[:batch_size]
            single_labels = labels[:batch_size]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'y_pred': preds,
            'y': labels,
            'y_single_pred': single_preds,
            'y_single': single_labels,
        }

    def transform_single_acc(y_pred, y_true):
        return (y_pred, y_true[:y_pred.shape[0]])

    engine = Engine(_update)
    engine.name = name
    metrics.Average(lambda o: o['loss']).attach(engine, 'loss')
    metrics.Accuracy(lambda o: (o['y_pred'], o['y'])).attach(engine, 'acc')
    metrics.Accuracy(lambda o: (o['y_single_pred'], o['y_single'])).attach(engine, 'single acc')

    return engine

def create_supervised_aug_evaluator(model,
                                    transform,
                                    joint=True,
                                    name='train',
                                    device=None):

    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.eval()

        with torch.no_grad():
            images, labels = convert_tensor(batch, device=device)
            batch_size = images.shape[0]
            images = transform(model, images)
            n = images.shape[0] // batch_size

            if joint:
                preds = model(images).view(n, batch_size, -1, n)
                mean_preds = 0
                for i in range(n):
                    mean_preds = mean_preds + preds[i, :, :, i] / n
                preds = preds[0, :, :, 0]
                loss = F.cross_entropy(mean_preds, labels)
            else:
                preds = model(images).view(n, batch_size, -1)
                mean_preds = preds.mean(0)
                preds = preds[0]
                loss = F.cross_entropy(mean_preds, labels)

        return {
            'loss': loss.item(),
            'y_pred': mean_preds,
            'y_single_pred': preds,
            'y': labels
        }

    engine = Engine(_update)
    engine.name = name
    metrics.Average(lambda o: o['loss']).attach(engine, 'loss')
    metrics.Accuracy(lambda o: (o['y_pred'], o['y'])).attach(engine, 'acc')
    metrics.Accuracy(lambda o: (o['y_single_pred'], o['y'])).attach(engine, 'single acc')
    return engine

def create_supervised_aug_sd_trainer(model,
                                     transform,
                                     optimizer,
                                     beta=1.0,
                                     name='train',
                                     device=None):

    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.train()

        images, labels = convert_tensor(batch, device=device)
        batch_size = images.shape[0]
        images = transform(model, images, labels)
        n = images.shape[0] // batch_size

        labels = torch.cat([labels*n+i for i in range(n)], 0)
        preds, single_preds = model(images, None)
        single_preds = single_preds[:batch_size]
        loss = F.cross_entropy(preds, labels)
        loss = loss + F.cross_entropy(single_preds, labels[:batch_size] // n).mul(beta)

        preds_reshaped = preds.view(n, batch_size, -1, n)
        mean_preds = 0
        for i in range(n):
            mean_preds = mean_preds + preds_reshaped[i, :, :, i] / n
        kl_div = F.kl_div(F.log_softmax(single_preds, 1),
                          F.softmax(mean_preds.detach(), 1),
                          reduction='batchmean')
        loss = loss + kl_div

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
    metrics.Average(lambda o: o['loss']).attach(engine, 'loss')
    metrics.Accuracy(lambda o: (o['y_pred'], o['y'])).attach(engine, 'acc')
    return engine

def create_supervised_aug_sd_evaluator(model,
                                       transform,
                                       name='train',
                                       device=None):

    if device is not None:
        model.to(device)

    def _update(engine, batch):
        model.eval()

        with torch.no_grad():
            images, labels = convert_tensor(batch, device=device)
            batch_size = images.shape[0]
            images = transform(model, images)
            n = images.shape[0] // batch_size

            preds, single_preds = model(images, None)
            preds = preds.view(n, batch_size, -1, n)
            single_preds = single_preds[:batch_size]
            mean_preds = 0
            for i in range(n):
                mean_preds = mean_preds + preds[i, :, :, i] / n
            loss = F.cross_entropy(mean_preds, labels)

        return {
            'loss': loss.item(),
            'y_agg_pred': mean_preds,
            'y_pred': single_preds,
            'y': labels
        }

    engine = Engine(_update)
    engine.name = name
    metrics.Average(lambda o: o['loss']).attach(engine, 'loss')
    metrics.Accuracy(lambda o: (o['y_pred'], o['y'])).attach(engine, 'acc')
    metrics.Accuracy(lambda o: (o['y_agg_pred'], o['y'])).attach(engine, 'agg acc')
    return engine
