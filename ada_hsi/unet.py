import torch
import torch.nn as nn
import torch.nn.functional as F


class convBlock(nn.Module):
    """(Conv2D => Batchnom => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(convBlock, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels

        conv_kwargs = dict(kernel_size=3, padding='same', bias=False)

        self.conv_block =  nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, **conv_kwargs),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):        
        return self.conv_block(x)


class downBlock(nn.Module):
    """Spatial downsampling and then convBlock"""

    def __init__(self, in_channels, out_channels):
        super(downBlock, self).__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            convBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class upBlock(nn.Module):
    """Spatial upsampling and then convBlock"""

    def __init__(self, in_channels, out_channels):
        super(upBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = convBlock(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)

class outBlock(nn.Module):
    """Conv2D => Sigmoid"""

    def __init__(self, in_channels, out_channels):
        super(outBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        return self.act( self.conv(x) )


class Unet(nn.Module):

    def __init__(self, n_channels=3, n_classes=9, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        n_classes = 9
        levels = len(features)
        self.inc = convBlock(n_channels, features[0])
        self.downs = nn.ModuleList([downBlock(features[i], features[i+1]) for i in range(levels-2)])
        self.bottle = downBlock(features[-2], features[-1] // 2)
        self.ups = nn.ModuleList([upBlock(features[i+1], features[i] // 2) for i in range(levels-2, 0, -1)])
        self.ups.append(upBlock(features[1], features[0]))
        self.outc = outBlock(features[0], n_classes)

    def forward(self, x):       

        outputs = []
        x = self.inc(x)
        outputs.append(x)
        
        for down in self.downs:
            x = down(x)
            outputs.append(x)

        x = self.bottle(x)

        for up in self.ups:
            x = up(x, outputs.pop())
        
        return self.outc(x)