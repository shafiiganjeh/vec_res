import functools
import torch.nn as nn
import torch
import sys
sys.path.append("..")
from layers import discr as l

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc = 1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


#simplified u-net discriminator 
class Unet_disc(nn.Module):

    def __init__(self,res = (256,512),inp_filter =  1,nonlinearity = l.dsc_nonlinearity,kernel = 3,filters = [8,16,32,64,128,256],down_scale = [1,2,1,2,1,2]):

            r"""
            Args:
                res (int): input resolution.
                inp_filter (int):input channel size.
                nonlinearity (torch.nn.Module): nonlinearity.
                kernel (int): convolution kernel size.
                down_scale (list of int): down scale per filter.
                filters (list of int): filters with channel size.
            """

        super(Unet_disc, self).__init__()
        
        self.nonlinearity = nonlinearity
        self.dwn = nn.ModuleList()
        self.up = nn.ModuleList()
        self.length = len(filters)

        filters = [inp_filter] + filters
        
        Yres = res[1]
        Xres = res[0]
        
        padX = []
        padY = []
        
        rev_filter_in = [i*2 for i in filters[:-1]]
        rev_filter_in = rev_filter_in + [filters[-1]]
        
        for i in range(self.length):
            if down_scale[i] > 1:

                if res[0] % 2:
                    padX.append(0)
                else:
                    padX.append(1)
                if res[1] % 2:
                    padY.append(0)
                else:
                    padY.append(1)
            
                self.dwn.append(torch.nn.utils.spectral_norm(torch.nn.Conv2d(filters[i],filters[i+1],kernel_size=kernel,stride=down_scale[i],padding=(padX[-1],padY[-1]))))
            else:
                self.dwn.append(torch.nn.utils.spectral_norm(torch.nn.Conv2d(filters[i],filters[i+1],kernel_size=kernel,stride=down_scale[i],padding="same")))
                padX.append(None)
                padY.append(None)
                
            Yres = Yres // 2
            Xres = Xres // 2
            
        for j in reversed(range(self.length)):

            if down_scale[j] > 1:
                self.up.append(torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(rev_filter_in[j+1],filters[j],kernel_size=kernel+1,stride=down_scale[j],padding = (padX[j],padY[j]))))
            else:
                self.up.append(torch.nn.utils.spectral_norm(torch.nn.Conv2d(rev_filter_in[j+1],filters[j],kernel_size=kernel,stride=down_scale[j],padding="same")))
        
        
        self.conv_low = torch.nn.Conv2d(filters[-1],
                                        1,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        h = x
        h1 = []
        Hi = []
        
        for i in range(self.length):
            h = self.nonlinearity(h)
            h = self.dwn[i](h)
            if self.length > i + 1:
                h1.append(h)
            
        h_low = self.conv_low(h)
        h1.append(None)
            
        for i in range(self.length):
            if h1[-(i+1)] is not None:
                h = torch.concat((h1[-(i+1)],h),dim = 1)
                
            h = self.nonlinearity(h)
            h = self.up[i](h)
            
            Hi.append(h)
            
        Hi = Hi[:-1]
            
        return h,h_low,Hi
    
#standard u-net discriminator
class simple_UNet(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, bilinear=False):

            r"""
            Args:
                n_classes (int):output channel size.
                n_channels (int):input channel size.
                bilinear (bool):bilinear downsampling.
            """

        super(simple_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (l.DoubleConv(n_channels, 8))
        self.down1 = (l.Down(8, 16))
        self.down2 = (l.Down(16, 32))
        self.down3 = (l.Down(32, 64))
        factor = 2 if bilinear else 1
        self.down4 = (l.Down(64, 128 // factor))
        self.up1 = (l.Up(128, 64 // factor, bilinear))
        self.up2 = (l.Up(64, 32 // factor, bilinear))
        self.up3 = (l.Up(32, 16 // factor, bilinear))
        self.up4 = (l.Up(16, 8, bilinear))
        self.outc = (l.OutConv(8, n_classes))

    def forward(self, x):
        Hi = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        h_low = x5
        x = self.up1(x5, x4)
        Hi.append(x)
        x = self.up2(x, x3)
        Hi.append(x)
        x = self.up3(x, x2)
        Hi.append(x)
        x = self.up4(x, x1)
        Hi.append(x)
        h = self.outc(x)
        return h,h_low,Hi
