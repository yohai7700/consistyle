from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# from appearance_transfer_model import AppearanceTransferModel
# from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_latents_or_invert_images(model, cfg):

    if all([cfg.load_latents,cfg.img1_latent_save_path.exists() ,cfg.img2_latent_save_path.exists(),
           cfg.edges1_latent_save_path.exists(), cfg.edges2_latent_save_path.exists(), cfg.style_latent_save_path.exists()]):

    # if cfg.load_latents and cfg.img1_latent_save_path.exists() and cfg.img2_latent_save_path.exists() and cfg.style_latent_save_path.exists():
        print("Loading existing latents...")
        latents = load_latents(cfg.img1_latent_save_path)
        noises = load_noise(cfg.img1_latent_save_path)
        print("Done.")
    else:
        print("Inverting images...")

        image_q1  = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents ,noises  = invert_images(image_q1=image_q1, sd_model=model.pipe, cfg=cfg)
        model.enable_edit = True
        print("Done.")
    return latents, noises

def load_latents(img1_latent_save_path: Path) -> Tuple[torch.Tensor]:

    latents_q1_im = torch.load(img1_latent_save_path)

    if type(latents_q1_im) == list:
        latents_q1_im = [l.to(device) for l in latents_q1_im]

    else:
        latents_q1_im = latents_q1_im.to(device)

    return latents_q1_im
   

def load_noise(img1_latent_save_path: Path) -> torch.Tensor:
    
    noise_q1_im = torch.load(img1_latent_save_path.parent / (img1_latent_save_path.stem + "_ddpm_noise.pt"))
    noise_q1_im = noise_q1_im.to(device)

    return  noise_q1_im


def invert_images(sd_model, image_q1: Image.Image, cfg):
    
    input_q1_im = torch.from_numpy(np.array(image_q1)).float() / 127.5 - 1.0
    zs_q1_im, latents_q1_im = invert(x0=input_q1_im.permute(2, 0, 1).unsqueeze(0).to(device),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
    

    # Save the inverted latents and noises
    torch.save(latents_q1_im, cfg.latents_path / f"{cfg.image_frame_1_path.stem}.pt")
    torch.save(zs_q1_im, cfg.latents_path / f"{cfg.image_frame_1_path.stem}_ddpm_noise.pt")

    latents =  latents_q1_im
    noises = zs_q1_im

    return latents, noises


def get_init_latents_and_noises(model, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_q1_im.dim() == 4 and model.latents_q1_im.shape[0] > 1:
        model.latents_q1_im = model.latents_q1_im[cfg.skip_steps]

    # we init out1 and  out2 from edges map
    init_latents = torch.stack([model.latents_q1_edges,model.latents_q1_edges])
    init_zs = [model.zs_q1_edges[cfg.skip_steps:], model.zs_q1_edges[cfg.skip_steps:]]
    return init_latents, init_zs
