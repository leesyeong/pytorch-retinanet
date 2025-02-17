import torch.nn as nn

class FPN(nn.Module):
    def __init__(self, in_channels, feature_size, pyramid_levels):
        super(FPN, self).__init__()

        if len(in_channels) != 3:
            raise ValueError("Expected a 3-element tuple for 'in_channels'")

        self.pyramid_levels = pyramid_levels

        if 2 in pyramid_levels:
            self.P2_1 = nn.Conv2d(in_channels[0], feature_size, kernel_size=1, stride=1, padding=0)
            self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            self.P3_1 = nn.Conv2d(in_channels[1], feature_size, kernel_size=1, stride=1, padding=0)
            self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            self.P4_1 = nn.Conv2d(in_channels[2], feature_size, kernel_size=1, stride=1, padding=0)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            self.P5 = nn.Conv2d(in_channels[2], feature_size, kernel_size=3, stride=2, padding=1)

            if 6 in pyramid_levels:
                self.P6_1 = nn.ReLU()
                self.P6_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        else:
            # upsample C5 to get P5 from the FPN paper
            self.P5_1 = nn.Conv2d(in_channels[2], feature_size, kernel_size=1, stride=1, padding=0)
            self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            # add P5 elementwise to C4
            self.P4_1 = nn.Conv2d(in_channels[1], feature_size, kernel_size=1, stride=1, padding=0)
            self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
            self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            # add P4 elementwise to C3
            self.P3_1 = nn.Conv2d(in_channels[0], feature_size, kernel_size=1, stride=1, padding=0)
            self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

            # "P6 is obtained via a 3x3 stride-2 conv on C5"
            self.P6 = nn.Conv2d(in_channels[2], feature_size, kernel_size=3, stride=2, padding=1)

            # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
            self.P7_1 = nn.ReLU()
            self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        Ca, Cb, Cc = inputs

        if 2 in self.pyramid_levels:
            P4 = self.P4_1(Cc)
            P4_upsampled_x = self.P4_upsampled(P4)
            P4 = self.P4_2(P4)

            P3 = self.P3_1(Cb)
            P3 = P4_upsampled_x + P3
            P3_upsampled = self.P3_upsampled(P3)
            P3 = self.P3_2(P3)

            P2 = self.P2_1(Ca)
            P2 = P3_upsampled + P2
            P2 = self.P2_2(P2)

            P5 = self.P5(Cc)

            if 6 in self.pyramid_levels:
                P6 = self.P6_2(self.P6_1(P5))

                return [P2, P3, P4, P5, P6]
            else:
                return [P2, P3, P4, P5]

        else:
            P5 = self.P5_1(Cc)
            P5_upsampled_x = self.P5_upsampled(P5)
            P5 = self.P5_2(P5)

            P4 = self.P4_1(Cb)
            P4 = P5_upsampled_x + P4
            P4_upsampled_x = self.P4_upsampled(P4)
            P4 = self.P4_2(P4)

            P3 = self.P3_1(Ca)
            P3 = P3 + P4_upsampled_x
            P3 = self.P3_2(P3)

            P6 = self.P6(Cc)

            P7 = self.P7_1(P6)
            P7 = self.P7_2(P7)

            return [P3, P4, P5, P6, P7]