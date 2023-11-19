import numpy as np
import torch.nn.functional as F

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def normalize_01(img):
    min = np.min(img)
    max = np.max(img)
    
    img_norm = (img - min) / (max - min + 1e-8)
    #img_norm = img_norm * 2 - 1
    return img_norm

def modified_act_fn(x):
    return x + F.relu((x - 1)/2) + F.relu(-1*(x + 1)/2)
