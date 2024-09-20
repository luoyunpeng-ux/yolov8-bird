class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x