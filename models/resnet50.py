import torch
from torch import nn


class Bottleneck(nn.Module):
    extension = 4

    def __init__(self, inplanes, planes, stride, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.extension,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.extension)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplane = 64
        super(ResNet, self).__init__()

        self.block = block
        self.layers = layers

        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.make_layer(self.block, 64, layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.extension, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplane != planes * block.extension:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplane,
                    planes * block.extension,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ), nn.BatchNorm2d(planes * block.extension)
            )

        layers = []
        layers.append(block(self.inplane, planes, stride, downsample))
        self.inplane = planes * block.extension
        for _ in range(1, blocks):
            layers.append(block(self.inplane, planes, 1))

        return nn.Sequential(*layers)


def ResNet50(num_classes=7):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
