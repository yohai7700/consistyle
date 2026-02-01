import pathlib
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.curdir)
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
# from config import RunConfig
import torch 
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_images(cfg, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:

    image_query1 = load_size(cfg.image_frame_1_path)
    image_query2 = load_size(cfg.image_frame_2_path)
  
    edges_query1 = load_size(cfg.edges_frame_1_path)
    edges_query2 = load_size(cfg.edges_frame_2_path)  
    style_image = load_size(cfg.style_image_path)

    if save_path is not None:
        Image.fromarray(image_query1).save(save_path / f"image_query1.png")


    return image_query1, edges_query1, image_query2, edges_query2, style_image


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    import numpy as np
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image

def resize_mask(image_mask_path: Path, res=(64,64), interpolation=cv2.INTER_NEAREST, combine_masks=True):
    """
    resize a given mask
    
    Args:
        image_mask (numpy.ndarray): The binary mask (e.g., 512x512).
        res (tuple): resolution of the new mask (h,w)
        interpolation (cv2 interpolation type): a method for interpolating 
        combine_masks: wether to combine into one mask

    Returns:
        resized mask (e.g., 64x64).
    """
    start_mask = torch.zeros((res))
    end_mask = torch.zeros((res))
    h, w = res[0], res[1]

    for i in range(0,4):
        
        image_mask = load_size(image_mask_path[i])
        # Resize mask using nearest-neighbor interpolation
        resized_mask = cv2.resize(image_mask, (w, h), interpolation=interpolation)
        resized_mask =  torch.from_numpy(np.array(resized_mask)).float() / 255.0

        if combine_masks:
            start_mask[resize_mask >0] = 1.0

    for i in range(4,8):
        image_mask = load_size(image_mask_path[i])
        # Resize mask using nearest-neighbor interpolation
        resized_mask = cv2.resize(image_mask, (w, h), interpolation=interpolation)
        resized_mask =  torch.from_numpy(np.array(resized_mask)).float() / 255.0

        if combine_masks:
            end_mask[resize_mask >0] = 1.0


    return start_mask[:,:,0], end_mask[:,:,0]

