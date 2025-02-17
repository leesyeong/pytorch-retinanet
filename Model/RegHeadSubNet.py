import torch.nn as nn

class RegHeadSubNet(nn.Module):
    def __init__(self, num_features_in, feature_size=256, num_anchors=9, num_regress=4 ):
        super(RegHeadSubNet, self).__init__()

        self.num_anchors = num_anchors
        self.num_regress = num_regress

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, self.num_anchors * self.num_regress , kernel_size=3, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.output(x)

        out = x.permute(0, 2, 3, 1)
        return out.reshape(x.shape[0], -1, self.num_regress)