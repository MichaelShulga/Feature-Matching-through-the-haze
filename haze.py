import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def set_visibility(depth_map, visibility):
    assert(0 <= visibility <= 1)
    d = depth_map
    d = d / d.max()
    d = d + ((d.max() - d.min()) / visibility - (d.max() + d.min())) / 2
    d = d / d.max()
    return np.clip(d, 0, 1)

def make_hazed(img_ideal, atmospheric_light, beta, depth_map):
    assert(img_ideal.dtype == np.uint8)
    J = img_ideal.astype(np.float32) / 255.0
    d = depth_map
    t = np.exp(-beta * d)
    I = J * t[..., None] + atmospheric_light * (1 - t[..., None])
    return (I * 255).clip(0, 255).astype(np.uint8)

def prepare_mask(mask, size, sigma=0.05):
    return gaussian_filter(cv2.resize(mask, size), sigma=sigma*size[0])
