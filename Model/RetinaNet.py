import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

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

    @staticmethod
    def from_string(s):
        try:
            return DetectMode[s]
        except KeyError:
            raise ValueError()

class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_regress=4, feature_size=256, mode:DetectMode=DetectMode.NORMAL):
        super(RetinaNet, self).__init__()

        self.mode = mode
        self.backbone = ResNet(pretrained = True)

        self.pyramid_levels = self.getPyramidLevels()

        self.in_channels = self.backbone.getOutChannels(self.pyramid_levels[:3])

        self.anchors = Anchors(pyramid_levels=self.pyramid_levels)
        self.fpn = FPN(self.in_channels, feature_size, self.pyramid_levels)
        self.cls_head = ClsHeadSubNet(feature_size, feature_size, len(self.anchors), num_classes)
        self.reg_head = RegHeadSubNet(feature_size, feature_size, len(self.anchors), num_regress)
        self.losses = FocalLoss()


    def forward(self, x):
        if self.training:
            img_batch, annotations = x
        else:
            img_batch = x

        C2, C3, C4, C5 = self.backbone(img_batch)

        if self.mode == DetectMode.NORMAL:
            features = self.fpn([C3, C4, C5])
        else:
            features = self.fpn([C2, C3, C4])

        cls_output = torch.cat([self.cls_head(f) for f in features], dim=1)
        reg_output = torch.cat([self.reg_head(f) for f in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.losses(cls_output, reg_output, anchors, annotations)
        else:
            return self.predict(img_batch, cls_output, reg_output, anchors)

    def regressBoxes(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * 0.1
        dy = deltas[:, :, 1] * 0.1
        dw = deltas[:, :, 2] * 0.2
        dh = deltas[:, :, 3] * 0.2

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes

    def clipBoxes(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes

    def predict(self, img_batch, cls_output, reg_output, anchors):
        transformed_anchors = self.regressBoxes(anchors, reg_output)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(cls_output.shape[2]):
            scores = torch.squeeze(cls_output[:, :, i])
            scores_over_thresh = (scores > 0.001)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

    # Hardcoded pyramid levels for each mode
    def getPyramidLevels(self, mode:DetectMode=None):
        pyramid_levels = []
        if mode == DetectMode.NORMAL:
            pyramid_levels = [3, 4, 5, 6, 7]
        elif mode == DetectMode.SMALL:
            pyramid_levels = [2, 3, 4, 5, 6]
        elif mode == DetectMode.SMALL_LITE:
            pyramid_levels = [2, 3, 4, 5]
        elif mode == None and self.mode != None:
            pyramid_levels = self.getPyramidLevels(self.mode)
        else:
            raise ValueError("Invalid mode")

        return pyramid_levels
