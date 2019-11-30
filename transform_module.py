"""
Pytorch image transform modules
"""
import torch
import torch.nn as nn
from transform_functional import *

class Rotation(nn.Module):
    def __init__(self, alpha=1):
        super(Rotation, self).__iRotation_()
        self.alpha = alpha

    def forward(self, x, angle):     
        return rotation(x, angle*self.alpha)
    
class SheerX(nn.Module):
    def __init__(self, alpha=1):
        super(SheerX, self).__init__()
        self.alpha = alpha

    def forward(self, x, c):     
        return sheer_x(x, c*self.alpha)
    
class SheerY(nn.Module):
    def __init__(self, alpha=1):
        super(SheerY, self).__init__()
        self.alpha = alpha

    def forward(self, x, c):     
        return sheer_y(x, c*self.alpha)
    
class TranslateX(nn.Module):
    def __init__(self, alpha=1):
        super(TranslateX, self).__init__()
        self.alpha = alpha

    def forward(self, x, c):     
        return translate_x(x, c*self.alpha)
    
class TranslateY(nn.Module):
    def __init__(self, alpha=1):
        super(TranslateY, self).__init__()
        self.alpha = alpha

    def forward(self, x, c):     
        return translate_y(x, c*self.alpha)
    
class Brightness(nn.Module):
    def __init__(self, alpha=0.5):
        super(Brightness, self).__init__()
        self.alpha = alpha

    def forward(self, x, b):     
        return brightness(x, b*self.alpha)
    
class Contrast(nn.Module):
    def __init__(self, alpha=1):
        super(Contrast, self).__init__()
        self.alpha = alpha

    def forward(self, x, b):     
        return contrast(x, b*self.alpha)  

    