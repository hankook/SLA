import torch, math
from torch import nn
from torch.nn import functional as F

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class PredictionModel(nn.Module):
    def __init__(self, model, classifiers):
        super(PredictionModel, self).__init__()
        self.base_model = model
        self.classifiers = nn.ModuleList(classifiers)

    def forward(self, x, idx=0):
        features = self.base_model(x)
        if type(idx) is int:
            return self.classifiers[idx](features)
        else:
            if idx is None:
                idx = list(range(len(self.classifiers)))
            return [self.classifiers[i](features) for i in idx]
