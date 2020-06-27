import torch
from torch import nn
from torch.nn import functional as F

def rotation():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    return _transform, 4

def rotation2():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 2]], 1).view(-1, *size)

    return _transform, 2

def color_perm():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        images = torch.stack([images,
                              torch.stack([images[:, 0, :, :], images[:, 2, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 0, :, :], images[:, 2, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 1, :, :], images[:, 0, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 6

def color_perm3():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        images = torch.stack([images,
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 3

def rot_color_perm6():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        out = []
        for x in [images, torch.rot90(images, 2, (2, 3))]:
            out.append(x)
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 6

def rot_color_perm12():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        out = []
        for k in range(4):
            x = torch.rot90(images, k, (2, 3))
            out.append(x)
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 12

def rot_color_perm24():
    def _transform(model, images, labels=None):
        size = images.shape[1:]
        out = []
        for k in range(4):
            x = torch.rot90(images, k, (2, 3))
            out.append(x)
            out.append(torch.stack([x[:, 0, :, :], x[:, 2, :, :], x[:, 1, :, :]], 1))
            out.append(torch.stack([x[:, 1, :, :], x[:, 0, :, :], x[:, 2, :, :]], 1))
            out.append(torch.stack([x[:, 1, :, :], x[:, 2, :, :], x[:, 0, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 0, :, :], x[:, 1, :, :]], 1))
            out.append(torch.stack([x[:, 2, :, :], x[:, 1, :, :], x[:, 0, :, :]], 1))
        return torch.stack(out, 1).view(-1, *size).contiguous()

    return _transform, 24
