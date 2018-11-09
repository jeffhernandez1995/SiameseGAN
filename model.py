import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import ContrastiveLoss

class generator(nn.Module):
    """
    Generator based on DCGAN (Deep convolution GAN)
    """
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0) #(B,100,1,1) > (B,128*8,4,4)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1) #(B,128*8,4,4) > (B,128*4,8,8)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1) #(B,128*4,8,8) > (B,128*2,16,16)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #(B,128*2,16,16) > (B,128,32,32)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1) #(B,128,32,32) > (B,3,64,64)


    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        #print(x.size())
        return x

class discriminator(nn.Module):
    """docstring for discriminator"""
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, d, 4, 2, 1), # (B,1,64,64) > (B,64,32,32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d, d*2, 4, 2, 1), # (B,64,32,32) > (B,64*2,16,16)
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*2, d*4, 4, 2, 1), # (B,64*2,16,16) > (B,64*4,8,8)
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d*4, d*8, 4, 2, 1),  # (B,64*4,8,8) > (B,64*8,4,4)
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.fcn = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.Linear(1024, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10))#,
            #nn.Linear(10, 2))
    
    # forward method
    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) 
        output = self.fcn(output)
        return output

    def forward(self,  input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2