import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from models import cifar_resnet as cresnet
from .modules import Reshape, PredictionModel

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))

def check_model(name, *output_types):
    if name.startswith('resnet') or name.startswith('resnext') or name.startswith('wide_resnet'):
        model  = resnet.__dict__[name]()
        n = model.fc.in_features
        model = nn.Sequential(model.conv1,
                              model.bn1,
                              model.relu,
                              model.maxpool,
                              model.layer1,
                              model.layer2,
                              model.layer3,
                              model.layer4,
                              model.avgpool,
                              Reshape(-1))
        model.num_features = n

    elif name.startswith('cresnet'):
        model = cresnet.__dict__[name[1:]]()
        model = nn.Sequential(model.conv_1_3x3,
                              model.bn_1,
                              nn.ReLU(inplace=True),
                              model.stage_1,
                              model.stage_2,
                              model.stage_3,
                              model.avgpool,
                              Reshape(-1))
        model.num_features = 64

    if len(output_types) > 0:
        modules = []
        for out in output_types:
            if type(out) is int:
                modules.append(nn.Linear(model.num_features, out))
            else:
                raise Exception('out should be integer')

        model = PredictionModel(model, modules)

    return model
