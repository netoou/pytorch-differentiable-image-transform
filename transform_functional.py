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

def gaussiannoise(imgs, var):
    # only need variance of normal
    c, h, w = imgs.shape[-3:]
    normal = torch.distributions.normal.Normal(torch.tensor(0.0).to(var.device), var.view(-1))
    noise = normal.rsample((c, h, w)).permute(3,0,1,2)

    trans_imgs = imgs + noise
    
    return trans_imgs

def sharpeness(imgs, m):
    blur_imgs = linear_blur(imgs)
    
    mask = imgs - blur_imgs

    return imgs + mask * m[None][None].reshape(imgs.size(0), 1, 1, 1)

# non-differentiable w.r.t prarm
def solarize(imgs, v):
    b, c, w, h = imgs.size()
    # Invert all pixel values above a threshold
    mask = imgs > v.repeat(1, c*w*h).reshape(b, c, w, h)
    
    trans_imgs = imgs
    trans_imgs[mask] = 1.0 - trans_imgs[mask]
    
    return trans_imgs

# no-param functions
def invert(imgs):
    return 1 - imgs

def bluring(imgs):
    conv_filters = (torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]) / 9).repeat(4,1).reshape(4,3,3).to(imgs.device)
    trans_imgs = []
    for img, c_filter in zip(imgs, conv_filters):
        trans_imgs.append(F.conv2d(img[None], c_filter.repeat(3,1,1,1), padding=0, groups=3,))
        
    bg = imgs.clone()
    trans_imgs = torch.cat(trans_imgs, dim=0)
    trans_imgs = torch.clamp(trans_imgs, 0, 1)
    
    bg[:,:, 1:-1, 1:-1] = trans_imgs
    
    return bg

def horizontal_flip(imgs):
    
    gridmat = torch.tensor([[-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],]).to(imgs.device)
    gridmat = gridmat.repeat(imgs.size(0), 1, 1)
    
    # Flip the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid, )
    
    return trans_imgs

def vertical_flip(imgs):
    
    gridmat = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],]).to(imgs.device)
    gridmat = gridmat.repeat(imgs.size(0), 1, 1)
    
    # Flip the original image
    grid = F.affine_grid(gridmat, imgs.size())
    trans_imgs = F.grid_sample(imgs, grid, )
    
    return trans_imgs


