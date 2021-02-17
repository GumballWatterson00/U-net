import torch.nn.functional as F
from .unet import *


class Unet(nn.Module):
    
  def __init__(self, n_channels, n_classes):
    super(Unet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes

    self.start = DoubleConv(n_channels, 64)
    self.down_1 = DownSampling(64, 128)
    self.down_2 = DownSampling(128, 256)
    self.down_3 = DownSampling(256, 512)
    self.down_4 = DownSampling(512, 1024)
    self.up_1 = UpSampling(1024, 512)
    self.up_2 = UpSampling(512, 256)
    self.up_3 = UpSampling(256, 128)
    self.up_4 = UpSampling(128, 64)
    self.finish = Output(64, n_classes)
  
  def forward(self, x):
    x1 = self.start(x)
    x2 = self.down_1(x1)
    x3 = self.down_2(x2)
    x4 = self.down_3(x3)
    x5 = self.down_4(x4)
    x = self.up_1(x5, x4)
    x = self.up_2(x, x3)
    x = self.up_3(x, x2)
    x = self.up_4(x, x1)
    pred_img = self.finish(x)
    return pred_img