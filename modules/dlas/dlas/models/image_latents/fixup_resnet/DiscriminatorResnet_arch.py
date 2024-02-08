import numpy as np
import torch
import torch.nn as nn

import dlas.torch_intermediary as ml

__all__ = ['FixupResNet', 'fixup_resnet18', 'fixup_resnet34',
           'fixup_resnet50', 'fixup_resnet101', 'fixup_resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.lrelu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.lrelu(out)

        return out


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.lrelu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.lrelu(out + self.bias2b)

        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.lrelu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_filters=64, num_classes=1000, input_img_size=64):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = num_filters
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = self._make_layer(block, num_filters, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, num_filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, num_filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, num_filters*8, layers[3], stride=2)
        self.bias2 = nn.Parameter(torch.zeros(1))
        reduced_img_sz = int(input_img_size / 32)
        self.fc1 = ml.Linear(
            num_filters * 8 * reduced_img_sz * reduced_img_sz, 100)
        self.fc2 = ml.Linear(100, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                    2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(
                        2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                    2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(
                    2 / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(
                        2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            '''
            elif isinstance(m, ml.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)'''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes *
                                 block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x + self.bias2)

        return x


def fixup_resnet18(**kwargs):
    """Constructs a Fixup-ResNet-18 model.2
    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def fixup_resnet34(**kwargs):
    """Constructs a Fixup-ResNet-34 model.
    """
    model = FixupResNet(FixupBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet50(**kwargs):
    """Constructs a Fixup-ResNet-50 model.
    """
    model = FixupResNet(FixupBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet101(**kwargs):
    """Constructs a Fixup-ResNet-101 model.
    """
    model = FixupResNet(FixupBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def fixup_resnet152(**kwargs):
    """Constructs a Fixup-ResNet-152 model.
    """
    model = FixupResNet(FixupBottleneck, [3, 8, 36, 3], **kwargs)
    return model


__all__ = ['FixupResNet', 'fixup_resnet18', 'fixup_resnet34',
           'fixup_resnet50', 'fixup_resnet101', 'fixup_resnet152']
