import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.resnet = None
        if pretrained:
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        else:
            self.resnet = torchvision.models.resnet50()

        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        C2 = self.layer1(x)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        return C2, C3, C4, C5

    def getOutChannels(self, pyramid_levels):
        result = []
        for level in pyramid_levels:
            if level == 2:
                result.append(self.layer1[-1].conv3.out_channels)
            if level == 3:
                result.append(self.layer2[-1].conv3.out_channels)
            if level == 4:
                result.append(self.layer3[-1].conv3.out_channels)
            if level == 5:
                result.append(self.layer4[-1].conv3.out_channels)
        return result
