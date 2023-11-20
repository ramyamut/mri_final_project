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
        self.final_layer = nn.Conv2d(channels[-1]*2, 2, kernel_size=3, padding='same')
    
    def forward(self, freq0, img0):
        freq = freq0
        img = img0
        for l in range(len(self.layers)-1):
            freq, img = self.layers[l](freq, img, freq0, img0)
        freq = self.layers[-1](freq, img, freq0, img0) # [B, 2, C_out, H, W]

        B, _, C, H, W = freq.shape
        freq = freq.reshape(B, 2*C, H, W) # [B, 2*C_out, H, W]
        freq = self.final_layer(freq) # [B, 2, H, W]
        freq = torch.complex(freq[:,0], freq[:,1]).unsqueeze(1) # [B, 1, H, W]
        out = torch.fft.ifft2(freq).real
        out = F.relu(out)
        return out
        

class InterleavedLayer(nn.Module):
    """
    Interleaved layer
    """
    def __init__(self, in_ch, out_ch, include_img=True):
        super(InterleavedLayer, self).__init__()
        self.include_img = include_img
        self.alpha = nn.Parameter(torch.zeros(size=(1,)))
        self.bn_freq = nn.BatchNorm2d(2*in_ch)
        self.conv_freq = nn.Conv2d(2*in_ch, 2*out_ch, kernel_size=3, padding='same')
        
        if self.include_img:
            self.beta = nn.Parameter(torch.zeros(size=(1,)))
            self.bn_img = nn.BatchNorm2d(in_ch)
            self.conv_img = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='same')
    
    def forward(self, freq, img, freq0, img0):
        # freq [B, 2, C_in, H, W], freq0 [B, 2, 1, H, W]
        # img [B, C_in, H, W], img0 [B, 1, H, W]
        B, _, Cin, H, W = freq.shape
        freq_complex = torch.complex(freq[:,0], freq[:,1]) # [B, C_in, H, W]
        alpha_mix = torch.sigmoid(self.alpha)
        freq_mix = (1 - alpha_mix) * torch.fft.fft2(img) # [B, C_in, H, W]
        freq_mix = torch.stack([freq_mix.real, freq_mix.imag], dim=1) # [B, 2, C_in, H, W]
        freq_mix = alpha_mix * freq + freq_mix # [B, 2, C_in, H, W]
        freq_mix = freq_mix.reshape(B, 2*Cin, H, W) # [B, 2*C_in, H, W]
        
        freq_norm = self.bn_freq(freq_mix) # [B, 2*C_in, H, W]
        
        freq_out = modified_act_fn(self.conv_freq(freq_norm)) # [B, 2*C_out, H, W]
        Cout = freq_out.shape[1] // 2
        freq_out = freq_out.reshape(B, 2, Cout, H, W) + freq0 # [B, 2, C_out, H, W]
        
        if self.include_img:
            beta_mix = torch.sigmoid(self.beta)
            img_mix = beta_mix * img + (1 - beta_mix) * torch.fft.ifft2(freq_complex).real # [B, C_in, H, W]
            img_norm = self.bn_img(img_mix) # [B, C_in, H, W]
            img_out = F.relu(self.conv_img(img_norm)) + img0 # [B, C_out, H, W]
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
        self.conv_img = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding='same')
        self.bn_freq = nn.BatchNorm2d(2*in_ch)
        self.conv_freq = nn.Conv2d(2*in_ch, 2*out_ch, kernel_size=3, padding='same')
    
    def forward(self, freq, _, freq0, img0):
        # freq [B, 2, C_in, H, W], freq0 [B, 2, 1, H, W]
        # img0 [B, 1, H, W]
        B, _, Cin, H, W = freq.shape
        
        freq_norm = freq.reshape(B, 2*Cin, H, W) # [B, 2*C_in, H, W]
        freq_norm = self.bn_freq(freq_norm) # [B, 2*C_in, H, W]
        freq_conv = modified_act_fn(self.conv_freq(freq_norm)) # [B, 2*C_out, H, W]
        Cout = freq_conv.shape[1] // 2
        freq_conv = freq_conv.reshape(B, 2, Cout, H, W) + freq0 # [B, 2, C_out, H, W]
        freq_complex = torch.complex(freq_conv[:,0], freq_conv[:,1]) # [B, C_out, H, W]
        img_out = torch.fft.ifft2(freq_complex).real # [B, C_out, H, W]
        
        img_norm = self.bn_img(img_out) # [B, C_out, H, W]
        img_conv = F.relu(self.conv_img(img_norm)) + img0 # [B, C_out, H, W]
        freq_out = torch.fft.fft2(img_conv) # [B, C_out, H, W]
        freq_out = torch.stack([freq_out.real, freq_out.imag], dim=1) # [B, 2, C_out, H, W]
        
        if self.include_img:
            return freq_out, img_out
        else:
            return freq_out
    
