import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoubleConv(nn.Module):
    
  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(DoubleConv, self).__init__()
    if not mid_channels:
        mid_channels = out_channels
    self.doubleconv = nn.Sequential( 
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.doubleconv(x)


class DownSampling(nn.Module):
      
  def __init__(self, in_channels, out_channels):
    super(DownSampling, self).__init__()
    self.maxpooling_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )
  
  def forward(self, x):
    return self.maxpooling_conv(x)


class UpSampling(nn.Module):
      
  def __init__(self, in_channels, out_channels):
    super(UpSampling, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    # Copy and crop
    x1 = self.up(x1)
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1) 
    return self.conv(x)


class Output(nn.Module):
    
  def __init__(self, in_channels, out_channels):
    super(Output, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)