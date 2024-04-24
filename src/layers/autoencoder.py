import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

    
class AttnBlock(nn.Module):
    def __init__(self, in_channels,mod = False):
        super().__init__()
        self.in_channels = in_channels
        self.mod = mod
        
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        b,c,h,w = x.shape
        
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        scale_factor = (int(c)**(-0.5))
        
        if self.mod:
            
            q = q.reshape(b,c,h*w)
            q = q.permute(0,2,1)
            k = k.reshape(b,c,h*w)
            k = k.permute(0,2,1)
            v = v.reshape(b,c,h*w)
            v = v.permute(0,2,1)
            
            h_ =  F.scaled_dot_product_attention(q,k,v,scale = scale_factor)
            h_ = h_.permute(0,2,1)
            h_ = h_.reshape(b,c,h,w)
        else:
            q = q.reshape(b,c,h*w)
            q = q.permute(0,2,1)  
            k = k.reshape(b,c,h*w) 
            w_ = torch.bmm(q,k)    
            w_ = w_ * scale_factor
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b,c,h*w)
            w_ = w_.permute(0,2,1)   
            h_ = torch.bmm(v,w_)    
            h_ = h_.reshape(b,c,h,w)
            
        h_ = self.proj_out(h_)
        return x+h_


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,se_act = False):
        super().__init__()
        self.in_channels = in_channels
        self.se_act = se_act
        
        if se_act:
            if in_channels == out_channels:
                self.se = tv.SqueezeExcitation(input_channels = in_channels,squeeze_channels = in_channels//16)
            else:
                self.se = tv.SqueezeExcitation(input_channels = out_channels,squeeze_channels = out_channels//16)
        
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.se_act:
            h = self.se(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
    
    
