import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_preds, reg_preds, anchors, gt_boxes):
        device = cls_preds.device
        batch_size = cls_preds.size(0)
        num_anchors = anchors.size(0)

        # IoU 계산
        ious = self.box_iou(anchors, gt_boxes)
        max_ious, max_ids = ious.max(dim=1)

        # Positive & Negative Anchor Mask
        pos_mask = max_ious >= 0.5
        neg_mask = max_ious < 0.4

        # Classification Target 생성
        cls_targets = torch.zeros_like(cls_preds, device=device)
        cls_targets[pos_mask] = 1

        # Regression Target 생성
        assigned_gt_boxes = gt_boxes[max_ids]
        reg_targets = self.encode_boxes(anchors, assigned_gt_boxes)

        # Focal Loss 계산 (Classification)
        alpha_factor = torch.where(cls_targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(cls_targets == 1, 1 - cls_preds, cls_preds)
        focal_weight = alpha_factor * focal_weight.pow(self.gamma)
        cls_loss = F.binary_cross_entropy(cls_preds, cls_targets, reduction='none')
        cls_loss = focal_weight * cls_loss
        cls_loss = cls_loss[pos_mask | neg_mask].sum() / batch_size

        # Smooth L1 Loss 계산 (Regression)
        reg_loss = F.smooth_l1_loss(reg_preds[pos_mask], reg_targets[pos_mask], reduction='sum')
        reg_loss = reg_loss / batch_size

        return cls_loss + reg_loss

    def box_iou(self, boxes1, boxes2):
        """
        IoU 계산 함수 (N, 4) vs (M, 4)
        boxes1, boxes2는 [x1, y1, x2, y2] 형식
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        union_area = area1[:, None] + area2 - inter_area
        return inter_area / union_area

    def encode_boxes(self, anchors, gt_boxes):
        """
        Anchor와 GT Box를 비교해 Regression Target 생성
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
        gt_ctr = gt_boxes[:, :2] + 0.5 * gt_wh

        delta_ctr = (gt_ctr - anchors_ctr) / anchors_wh
        delta_wh = torch.log(gt_wh / anchors_wh)

        return torch.cat([delta_ctr, delta_wh], dim=1)
