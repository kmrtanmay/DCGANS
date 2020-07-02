import torch
import torch.nn as nn

# Defining a convolution transpose block for Generator
def Gen_conv_block(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
      nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(True)
  )

# Defining the Generator model
class Generator(nn.Module):
  def __init__(self):
    super(Generator,self).__init__()
    self.model = nn.Sequential(
        # input is Z, going into a convolution
        Gen_conv_block(num_z,num_gf*8,kernel_size=4,stride=1,padding=0),
        # state size. (num_gf*8) x 4 x 4
        Gen_conv_block(num_gf*8,num_gf*4,kernel_size=4,stride=2,padding=1),
        # state size. (num_gf*4) x 8 x 8
        Gen_conv_block(num_gf*4,num_gf*2,kernel_size=4,stride=2,padding=1),
        # state size. (num_gf*2) x 16 x 16
        Gen_conv_block(num_gf*2,num_gf,kernel_size=4,stride=2,padding=1),
        # state size. (num_gf) x 32 x 32
        nn.ConvTranspose2d(num_gf,num_channels,kernel_size=4,stride=2,padding=1,bias=False),
        nn.Tanh()
        # state size. (num_channels) x 64 x 64
    )
  def forward(self,input):
    output = self.model(input)
    return output

# Defining a convolution block for Discriminator 
def Dis_conv_block(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
      nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(negative_slope=0.2,inplace=True)
  )

# Defining The Discriminator model
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.model = nn.Sequential(
        # input is (num_channels) x 64 x 64
        nn.Conv2d(num_channels,num_df,kernel_size=4,stride=2,padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (num_df) x 32 x 32
        Dis_conv_block(num_df,num_df*2,kernel_size=4,stride=2,padding=1),
        # state size. (ndf*2) x 16 x 16
        Dis_conv_block(num_df*2,num_df*4,kernel_size=4,stride=2,padding=1),
        # state size. (ndf*4) x 8 x 8
        Dis_conv_block(num_df*4,num_df*8,kernel_size=4,stride=2,padding=1),
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(num_df*8,out_channels=1,kernel_size=4,stride=1,padding=0, bias=False),
        nn.Sigmoid()
      )
  def forward(self,input):
    output = self.model(input)
    return output