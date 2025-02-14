import enum
import torch
import torch.nn as nn


from .ResNet import ResNet
from .ClsHeadSubNet import ClsHeadSubNet
from .RegHeadSubNet import RegHeadSubNet
from .Anchors import Anchors
from .FPN import FPN
from .FocalLoss import FocalLoss

class DetectMode(enum.Enum):
    NORMAL = 1
    SMALL = 2
    SMALL_LITE = 3

class RetinaNet(nn.Module):
    #
    def __init__(self, num_classes, num_regress=4, feature_size=256, mode:DetectMode=DetectMode.NORMAL):
        super(RetinaNet, self).__init__()

        self.backbone = ResNet(True)

        if mode == DetectMode.NORMAL:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        elif mode == DetectMode.SMALL:
            self.pyramid_levels = [2, 3, 4, 5, 6]
        elif mode == DetectMode.SMALL_LITE:
            self.pyramid_levels = [2, 3, 4, 5]

        self.in_channels = self.backbone.getOutChannels(self.pyramid_levels[:3])

        self.anchors = Anchors(pyramid_levels=self.pyramid_levels)
        self.fpn = FPN(self.in_channels, feature_size, self.pyramid_levels)
        self.cls_head = ClsHeadSubNet(feature_size, feature_size, len(self.anchors), num_classes)
        self.reg_head = RegHeadSubNet(feature_size, feature_size, len(self.anchors), num_regress)
        self.losses = FocalLoss()


    def forward(self, x):
        C2, C3, C4, C5 = self.resnet(x)
        features = self.fpn([C3, C4, C5])
        cls_output = torch.cat([self.cls_head(f) for f in features], dim=1)
        reg_output = torch.cat([self.reg_head(f) for f in features], dim=1)

        anchors = self.anchors(x[0]) # 0: image, 1: target

        if self.training:
            return self.losses(cls_output, reg_output, anchors, x[1])
        else:
            self.predict(cls_output, reg_output, anchors)