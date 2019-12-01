"""
Differential functions of pytorch image transform operations
To make differentiable function, I used pytorch tensor operations and functionals
"""
import torch.nn.functional as F
import torch

from utils import *

def rotation(imgs, angle):
    h, w = imgs.shape[2:]
    
    # Get differentiable transform matrix 
    sin = angle.sin()
    cos = angle.cos()
    zero = angle - angle
    gridmat1 = torch.cat((cos, -sin*h/w, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat2 = torch.cat((sin*h/w, cos, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat = torch.cat((gridmat1, gridmat2), dim=1)
    
    # Rotate the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid,)
    
    return trans_imgs

def sheer_x(imgs, c):
    h, w = imgs.shape[2:]
    
    # Get differentiable transform matrix
    one = c - c + 1
    zero = c - c
    gridmat1 = torch.cat((one, c*h/w, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat2 = torch.cat((zero, one, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat = torch.cat((gridmat1, gridmat2), dim=1)
    
    # Rotate the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid,)
    
    return trans_imgs

def sheer_y(imgs, c):
    h, w = imgs.shape[2:]
    
    # Get differentiable transform matrix
    one = c - c + 1
    zero = c - c
    gridmat1 = torch.cat((one, zero, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat2 = torch.cat((c*h/w, one, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat = torch.cat((gridmat1, gridmat2), dim=1)
    
    # Rotate the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid,)
    
    return trans_imgs

def translate_x(imgs, c):
    h, w = imgs.shape[2:]
    
    # Get differentiable transform matrix
    one = c - c + 1
    zero = c - c
    gridmat1 = torch.cat((one, zero, c), dim=1).reshape(imgs.size(0),1,3)
    gridmat2 = torch.cat((zero, one, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat = torch.cat((gridmat1, gridmat2), dim=1)
    
    # Rotate the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid,)
    
    return trans_imgs

def translate_y(imgs, c):
    h, w = imgs.shape[2:]

    # Get differentiable transform matrix
    one = c - c + 1
    zero = c - c
    gridmat1 = torch.cat((one, zero, zero), dim=1).reshape(imgs.size(0),1,3)
    gridmat2 = torch.cat((zero, one, c), dim=1).reshape(imgs.size(0),1,3)
    gridmat = torch.cat((gridmat1, gridmat2), dim=1)
    
    # Rotate the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid,)
    
    return trans_imgs

def brightness(imgs, b):
    c, h, w = imgs.shape[-3:]
    trans_imgs = torch.clamp(imgs + b.repeat(1, h*w*c).reshape(imgs.size()), 0, 1)
    
    return trans_imgs

def contrast(imgs, b):
    c, h, w = imgs.shape[-3:]
    trans_imgs = torch.clamp(imgs * (1 + b.repeat(1, h*w*c).reshape(imgs.size())), 0, 1)
    
    return trans_imgs

def saturation(imgs, s):
    c, h, w = imgs.shape[-3:]
    hsv_imgs = rgb_to_hsv(imgs)
    hue, sat, val = hsv_imgs[:, 0, :, :], hsv_imgs[:, 1, :, :], hsv_imgs[:, 2, :, :]
    sat = sat * s.repeat(1, h*w).reshape(-1, h, w)
    trans_imgs = torch.stack((hue, sat, val), dim=1)
    
    return hsv_to_rgb(trans_imgs)


