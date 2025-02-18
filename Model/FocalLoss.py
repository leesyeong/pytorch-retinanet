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
        num_anchors = anchors.size(1)

        # IoU 계산
        ious = self.box_iou(anchors, gt_boxes[..., :4])
        max_ious, max_ids = ious.max(dim=2)

        # Positive & Negative Anchor Mask
        pos_mask = max_ious > 0.5
        neg_mask = max_ious < 0.4

        # Classification Target 생성
        cls_targets = torch.zeros_like(cls_preds, device=device)
        cls_targets[pos_mask] = 1

        # Regression Target 생성
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1)
        # assigned_gt_boxes = torch.zeros((batch_size, num_anchors, 5), device=device)
        assigned_gt_boxes = gt_boxes[batch_indices, max_ids][pos_mask]
        polsitive_anchors = anchors.repeat(batch_size,1,1)[pos_mask]
        reg_targets = self.encode_boxes(polsitive_anchors, assigned_gt_boxes[..., :4])

        # Focal Loss 계산 (Classification)
        alpha_factor = torch.where(cls_targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(cls_targets == 1, 1 - cls_preds, cls_preds)
        focal_weight = alpha_factor * focal_weight.pow(self.gamma)
        cls_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_targets, reduction='none')
        cls_loss = focal_weight * cls_loss
        cls_loss = cls_loss[pos_mask | neg_mask].sum() / batch_size

        # Smooth L1 Loss 계산 (Regression)
        reg_loss = F.smooth_l1_loss(reg_preds[pos_mask], reg_targets, reduction='sum')
        reg_loss = reg_loss / batch_size

        return cls_loss, reg_loss

    def box_iou(self, anchors, gt_boxes):
        """
        Anchors: (1, N, 4) - (xmin, ymin, xmax, ymax)
        GT Boxes: (B, M, 4) - (xmin, ymin, xmax, ymax)
        Returns:
            iou: (B, N, M) - IoU between each anchor and each GT box
        """
        # anchors: (N, 4)
        bboxes1 = anchors.squeeze(0)

        # (B, N, M, 1) 형태로 Broadcast를 위한 차원 확장
        bboxes1 = bboxes1.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 4)
        bboxes2 = gt_boxes.unsqueeze(1)             # (B, 1, M, 4)

        # 좌표 추출
        anchor_xmin = bboxes1[..., 0]  # (B, N, M)
        anchor_ymin = bboxes1[..., 1]
        anchor_xmax = bboxes1[..., 2]
        anchor_ymax = bboxes1[..., 3]

        gt_xmin = bboxes2[..., 0]  # (B, N, M)
        gt_ymin = bboxes2[..., 1]
        gt_xmax = bboxes2[..., 2]
        gt_ymax = bboxes2[..., 3]

        # 교집합 좌표 계산
        inter_xmin = torch.maximum(anchor_xmin, gt_xmin)  # (B, N, M)
        inter_ymin = torch.maximum(anchor_ymin, gt_ymin)
        inter_xmax = torch.minimum(anchor_xmax, gt_xmax)
        inter_ymax = torch.minimum(anchor_ymax, gt_ymax)

        # 교집합 넓이 계산
        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h  # (B, N, M)

        # 개별 박스의 넓이 계산
        anchor_area = (anchor_xmax - anchor_xmin) * (anchor_ymax - anchor_ymin)  # (B, N, M)
        gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)                      # (B, N, M)

        # 합집합 넓이 계산
        union_area = anchor_area + gt_area - inter_area

        # IoU 계산
        iou = inter_area / union_area.clamp(min=1e-6)  # (B, N, M)

        return iou

    def encode_boxes(self, anchors, gt_boxes):
        """
        Anchor와 GT Box를 비교해 Regression Target 생성
        """
        anchors_wh = anchors[..., 2:] - anchors[..., :2]
        anchors_ctr = anchors[..., :2] + 0.5 * anchors_wh

        gt_wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
        gt_ctr = gt_boxes[..., :2] + 0.5 * gt_wh

        delta_ctr = (gt_ctr - anchors_ctr) / anchors_wh
        delta_wh = torch.log(gt_wh / anchors_wh)

        return torch.cat([delta_ctr, delta_wh], dim=1)
