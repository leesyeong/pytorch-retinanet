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
            P2 = self.P2_2(self.P2_1(Ca))
            P3 = self.P3_2(self.P3_upsampled(self.P3_1(Cb)))
            P4 = self.P4_2(self.P4_upsampled(self.P4_1(Cc)))
            P5 = self.P5(Cc)

            if 6 in self.pyramid_levels:
                P6 = self.P6_2(self.P6_1(P5))

                return [P2, P3, P4, P5, P6]
            else:
                return [P2, P3, P4, P5]

        else:
            P3 = self.P3_2(self.P3_1(Ca))
            P4 = self.P4_2(self.P4_upsampled(self.P4_1(Cb)))
            P5 = self.P5_2(self.P5_upsampled(self.P5_1(Cc)))
            P6 = self.P6(Cc)
            P7 = self.P7_2(self.P7_1(P6))

            return [P3, P4, P5, P6, P7]