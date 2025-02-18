import torch.nn as nn

class ClsHeadSubNet(nn.Module):
    def __init__(self, feature_in_size, feature_size, num_anchors, num_classes):
        super(ClsHeadSubNet, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(feature_in_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.cls_head = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.cls_head(x)
        x = self.out_activation(x)

        out = x.permute(0, 2, 3, 1)
        batch_size, height, width, channels = out.shape
        out = out.reshape(batch_size, width, height, self.num_anchors, self.num_classes)
        return out.reshape(x.shape[0], -1, self.num_classes)