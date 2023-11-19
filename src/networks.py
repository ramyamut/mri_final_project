import torch
from torch import nn
import torch.nn.functional as F

from src.utils import modified_act_fn

class Net(nn.Module):
    """
    Network architecture
    """
    def __init__(self, channels, layer_type):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        layer_fn = InterleavedLayer if layer_type == 'interleaved' else AlternatingLayer
        channels = [1] + channels
        for l in range(len(channels)-2):
            self.layers.append(layer_fn(channels[l], channels[l+1]))
        self.layers.append(layer_fn(channels[-2], channels[-1], include_img=False))
        self.final_layer = nn.Conv2d(channels[-1]*2, 2)
    
    def forward(self, freq0, img0):
        freq = freq0
        img = img0
        for l in range(len(self.layers)-1):
            freq, img = self.layers[l](freq, img, freq0, img0)
        freq = self.layers[-1](freq, img, freq0, img0)
        
        freq = freq.reshape(freq.shape[0], 2*freq.shape[2], -1)
        freq = self.final_layer(freq)
        freq = torch.complex(freq[:,0], freq[:,1])
        out = torch.fft.ifft2(freq).
        out = F.relu(out)
        return out
        

class InterleavedLayer(nn.Module):
    """
    Interleaved layer
    """
    def __init__(self, in_ch, out_ch, include_img=True):
        super(InterleavedLayer, self).__init__()
        self.include_img = include_img
        self.alpha = nn.Parameter(torch.zeros(size=(1,)).to(self.device))
        self.bn_freq = nn.BatchNorm2d(2*in_ch)
        self.conv_freq = nn.Conv2d(2*in_ch, 2*out_ch)
        
        if self.include_img:
            self.beta = nn.Parameter(torch.zeros(size=(1,)).to(self.device))
            self.bn_img = nn.BatchNorm2d(out_ch)
            self.conv_img = nn.Conv2d(in_ch, out_ch)
    
    def forward(self, freq, img, freq0, img0):
        # freq & freq0 have shape [B, 2, C_in, H, W]
        freq_complex = torch.complex(freq[:,0], freq[:,1])
        alpha_mix = torch.sigmoid(self.alpha)
        freq_mix = alpha_mix * freq_complex + (1 - alpha_mix) * torch.fft.fft2(img)
        freq_mix = freq_mix.reshape(freq_mix.shape[0], 2*freq_mix.shape[2], -1)
        
        
        freq_norm = self.bn_freq(freq_mix)
        
        freq_out = modified_act_fn(self.conv_freq(freq_norm))
        freq_out = freq_out.reshape(freq_out.shape[0], 2, -1) + freq0
        
        if self.include_img:
            beta_mix = torch.sigmoid(self.beta)
            img_mix = beta_mix * img + (1 - beta_mix) * torch.fft.ifft2(freq_complex).real
            img_norm = self.bn_img(img_mix)
            img_out = F.relu(self.conv_img(img_norm)) + img0
            return freq_out, img_out
        
        else:
            return freq_out
        
class AlternatingLayer(nn.Module):
    """
    Alternating layer
    """
    def __init__(self, in_ch, out_ch, include_img=True):
        super(AlternatingLayer, self).__init__()
        self.include_img = include_img
        self.bn_img = nn.BatchNorm2d(out_ch)
        self.conv_img = nn.Conv2d(in_ch, out_ch)
        
        if self.include_img:
            self.bn_freq = nn.BatchNorm2d(2*in_ch)
            self.conv_freq = nn.Conv2d(2*in_ch, 2*out_ch)
    
    def forward(self, freq, img, freq0, img0):
        
        img_norm = self.bn_img(img)
        img_conv = F.relu(self.conv_img(img_norm)) + img0
        freq_out = torch.fft.fft2(img_conv)
        freq_out = torch.stack([freq_out.real, freq_out.imag], dim=1)
        
        if self.include_img:
            freq_norm = freq.reshape(freq.shape[0], 2*freq.shape[2], -1)
            freq_norm = self.bn_freq(freq_norm)
            freq_conv = modified_act_fn(self.conv_freq(freq_norm)) + freq0
            freq_conv = freq_conv.reshape(freq_conv.shape[0], 2, -1)
            freq_complex = torch.complex(freq_conv[:,0], freq_conv[:,1])
            img_out = torch.fft.ifft2(freq_complex).real
            return freq_out, img_out
        else:
            return freq_out
        
    
