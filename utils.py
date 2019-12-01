import torch

def rgb_to_hsv(img, eps=1e-8):
    hsv_img = torch.zeros_like(img)
    
    r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    
    max_rgb, max_mask = img.max(dim=1)
    min_rgb, min_mask = img.min(dim=1)
    delta = max_rgb - min_rgb
    
    zero_delta_mask = max_rgb == min_rgb
    r_mask = (max_mask==0)
    g_mask = (max_mask==1)
    b_mask = (max_mask==2)
    
    # calculate hue
    # the hsv tensor is initialized to zero values
    # r
    hsv_img[:, 0, :, :][r_mask] = ((g[r_mask] - b[r_mask]) / (delta[r_mask] + eps) + 6) / 6 % 6
    # g
    hsv_img[:, 0, :, :][g_mask] = (((b[g_mask] - r[g_mask]) / (delta[g_mask] + eps)) + 2) / 6 % 6
    # b
    hsv_img[:, 0, :, :][b_mask] = (((r[b_mask] - g[b_mask]) / (delta[b_mask] + eps)) + 4) / 6 % 6
    # in delta == 0 case
    hsv_img[:, 0, :, :][delta == 0] = 0
    
    # calculate saturation
    hsv_img[:, 1, :, :][max_rgb!= 0] = delta[max_rgb!= 0] / (max_rgb[[max_rgb!= 0]] + eps)
    
    # calculate value
    hsv_img[:, 2, :, :] = max_rgb
    
    return hsv_img
    
def hsv_to_rgb(img, eps=1e-8):
    rgb_img = torch.zeros_like(img)
    
    h, s, v = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

    c = v * s
    x = c * (1 - ((h * 6) % 2 - 1).abs())
    m = v - c
    
    mask = h < (1/6)
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = c[mask], x[mask], 0
    
    mask = (h >= (1/6)) * (h < (2/6))
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = x[mask], c[mask], 0
    
    mask = (h >= (2/6)) * (h < (3/6))
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = 0, c[mask], x[mask]
    
    mask = (h >= (3/6)) * (h < (4/6))
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = 0, x[mask], c[mask]
    
    mask = (h >= (4/6)) * (h < (5/6))
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = x[mask], 0, c[mask]
    
    mask = (h >= (5/6)) * (h < (6/6))
    rgb_img[:, 0, :, :][mask], rgb_img[:, 1, :, :][mask], rgb_img[:, 2, :, :][mask] = c[mask], 0, x[mask]
    
    rgb_img[:, 0, :, :], rgb_img[:, 1, :, :], rgb_img[:, 2, :, :] = rgb_img[:, 0, :, :] + m, rgb_img[:, 1, :, :] + m, rgb_img[:, 2, :, :] + m
    
    return rgb_img