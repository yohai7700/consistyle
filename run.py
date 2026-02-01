# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
import torch
from diffusers import DDIMScheduler
from unet_sdxl import ConsistorySDXLUNet2DConditionModel
from pipeline import ConsistoryExtendAttnSDXLPipeline
from attn_utils import FeatureInjector, AnchorCache
from utils.general_utils import *
from utils.latent_utils import load_latents_or_invert_images
import gc
import numpy as np
from PIL import Image

from utils.ptp_utils import AttentionStore, view_images

LATENT_RESOLUTIONS = [32, 64]

def load_pipeline(gpu_id=0, float_type=torch.float16):
    
    sd_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    unet = ConsistorySDXLUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=float_type)
    scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")

    story_pipeline = ConsistoryExtendAttnSDXLPipeline.from_pretrained(
        sd_id, unet=unet, torch_dtype=float_type, variant="fp16", use_safetensors=True, scheduler=scheduler
    ).to(device)
    story_pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    
    return story_pipeline

def create_anchor_mapping(bsz, anchor_indices=[0]):
    anchor_mapping = torch.eye(bsz, dtype=torch.bool)
    for anchor_idx in anchor_indices:
        anchor_mapping[:, anchor_idx] = True

    return anchor_mapping

def create_token_indices(prompts, batch_size, concept_token, tokenizer):
    if isinstance(concept_token, str):
        concept_token = [concept_token]

    concept_token_id = [tokenizer.encode(x, add_special_tokens=False)[0] for x in concept_token]
    tokens = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors='pt')['input_ids']

    token_indices = torch.full((len(concept_token), batch_size), -1, dtype=torch.int64)
    for i, token_id in enumerate(concept_token_id):
        batch_loc, token_loc = torch.where(tokens == token_id)
        token_indices[i, batch_loc] = token_loc

    return token_indices

def create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type):
    # if seed is int
    if isinstance(seed, int):
        g = torch.Generator(device).manual_seed(seed)
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=float_type)
    elif isinstance(seed, list):
        shape = (batch_size, story_pipeline.unet.config.in_channels, 128, 128)
        latents = torch.empty(shape, device=device, dtype=float_type)
        for i, seed_i in enumerate(seed):
            g = torch.Generator(device).manual_seed(seed_i)
            curr_latent = randn_tensor(shape, generator=g, device=device, dtype=float_type)
            latents[i] = curr_latent[i]

    if same_latent:
        latents = latents[:1].repeat(batch_size, 1, 1, 1)

    return latents, g


class GenerationResult:
    def __init__(self, name: str, images, downscale_rate=2):
        self.images = images
        self.name = name
        self.image_all = view_images([np.array(x) for x in images], display_image=False, downscale_rate=downscale_rate)

    def save(self, out_dir):
        dir = f'{out_dir}/{self.name}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        for i, image in enumerate(self.images):
            image.save(f'{dir}/image_{i}.png')
        self.image_all.save(f'{dir}/all.png')
        
# Batch inference
def run_batch_generation(story_pipeline, prompts, concept_token,
                        seed=40, n_steps=50, mask_dropout=0.5,
                        same_latent=False,record_queries=False,
                        share_queries=True,
                        invert = False,
                        perform_sdsa=True, perform_consistory_injection=True,
                        perform_styled_injection=True,
                        downscale_rate=4, n_achors=2, 
                        background_adain=None,
                        perform_original_sdxl=True,
                        use_target_heads=False,
                        attn_v_range=[3,10], 
                        attn_qk_range=[5,15],
                        perform_ablations=False,
                        perform_adain_ablation=False,
                        perform_no_dift_ablation=False):
    
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)
    anchor_mappings = create_anchor_mapping(batch_size, anchor_indices=list(range(n_achors)))

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout,
        'extended_mapping': anchor_mappings
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs= {'t_range': [0,n_steps], 'strength_start': 0.9, 'strength_end': 0.81836735}

    # if invert:
    #     latents, noises = load_latents_or_invert_images(model=model, cfg=cfg)
    #     model.set_latents(latents)
    #     model.set_noise(noises)
    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    # ------------------ #
    # Extended attention First Run #

    results: List[GenerationResult] = []
    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    # SDXL original
    attnstore = AttentionStore(default_attention_store_kwargs)
    if perform_original_sdxl:
        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            callback_steps=n_steps,
                            perform_extend_attn=False,
                            record_values=True,
                            attn_v_range = attn_v_range,
                            attn_qk_range = attn_qk_range,
                            record_queries=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps)
        results.append(GenerationResult('original sdxl', out.images, downscale_rate=downscale_rate))
    
    # first pass
    print(extended_attn_kwargs['t_range'])
    out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        attn_qk_range=attn_qk_range,
                        query_store_kwargs=query_store_kwargs,
                        callback_steps=n_steps,
                        num_inference_steps=n_steps,
                        record_queries=False,
                        attnstore=attnstore)
    last_masks = story_pipeline.attention_store.last_mask

    dift_features = unet.latent_store.dift_features['261_0'][batch_size:]
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

    nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)
    results.append(GenerationResult('first pass', out.images, downscale_rate=downscale_rate))
    # results.append(GenerationResult('first pass masks 64', transform_masks_to_images(last_masks[64], batch_size), downscale_rate=downscale_rate))

    torch.cuda.empty_cache()
    gc.collect()

    # ------------------ #
    # Extended attention with nn_map #
    
    if perform_consistory_injection:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_consistory_feature_injection=True,
                            use_styled_feature_injection=False)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)
        results.append(GenerationResult('consistory', out.images, downscale_rate=downscale_rate))
        
        dift_masks = [feature_injector.get_nn_map(i, 64, anchor_mappings)[3] for i in range(batch_size)]
        # results.append(GenerationResult('consistory dift masks', transform_masks_to_images(dift_masks, batch_size), downscale_rate=downscale_rate))
        torch.cuda.empty_cache()
        gc.collect()
    
    if perform_styled_injection:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)
        results.append(GenerationResult(f'consistyle', out.images, downscale_rate=downscale_rate))
        
        # dift_masks = [feature_injector.get_nn_map(i, 64, anchor_mappings)[3] for i in range(batch_size)]
        # results.append(GenerationResult('consistyle dift masks', transform_masks_to_images(dift_masks, batch_size), downscale_rate=downscale_rate))
        
        # del attnstore
        torch.cuda.empty_cache()
        gc.collect()
        
    if perform_ablations:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range,
                            no_attn_q=True,
                            no_attn_k=True
                            )
        results.append(GenerationResult(f'consistyle_no_qk', out.images, downscale_rate=downscale_rate))
        torch.cuda.empty_cache()
        gc.collect()
        
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range,
                            no_attn_q=True,
                            )
        results.append(GenerationResult(f'consistyle_no_q', out.images, downscale_rate=downscale_rate))
        torch.cuda.empty_cache()
        gc.collect()
        
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range,
                            no_attn_k=True
                            )
        results.append(GenerationResult(f'consistyle_no_k', out.images, downscale_rate=downscale_rate))
        torch.cuda.empty_cache()
        gc.collect()
        
    if perform_adain_ablation:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range,
                            use_consistyle_adain=False)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)
        results.append(GenerationResult(f'consistyle_no_adain', out.images, downscale_rate=downscale_rate))
        
        # dift_masks = [feature_injector.get_nn_map(i, 64, anchor_mappings)[3] for i in range(batch_size)]
        # results.append(GenerationResult('consistyle dift masks', transform_masks_to_images(dift_masks, batch_size), downscale_rate=downscale_rate))
        
        # del attnstore
        torch.cuda.empty_cache()
        gc.collect()
        
    if perform_no_dift_ablation:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic', background_adain=background_adain, background_self_alignment_range=(n_steps//3 + 1, n_steps//3 + 2))

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            use_styled_feature_injection=True,
                            use_first_half_target_heads=use_target_heads,
                            use_consistory_feature_injection=False,
                            attnstore=attnstore,
                            num_inference_steps=n_steps,
                            attn_v_range=attn_v_range,
                            attn_qk_range=attn_qk_range,
                            use_consistyle_direct_injection=True)
        results.append(GenerationResult(f'consistyle_no_dift', out.images, downscale_rate=downscale_rate))
        
        torch.cuda.empty_cache()
        gc.collect()
        
    del attnstore
    torch.cuda.empty_cache()
    gc.collect()
    return results

# Anchors
def run_anchor_generation(story_pipeline, prompts, concept_token,
                        seed=40, n_steps=2, mask_dropout=0.5,
                        same_latent=False, share_queries=True,
                        perform_sdsa=True, perform_injection=True,
                        downscale_rate=4, cache_cpu_offloading=False):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    latents, g = create_latents(story_pipeline, seed, batch_size, same_latent, device, float_type)

    anchor_cache_first_stage = AnchorCache()
    anchor_cache_second_stage = AnchorCache()

    # ------------------ #
    # Extended attention First Run #

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        query_store_kwargs=query_store_kwargs,
                        anchors_cache=anchor_cache_first_stage,
                        num_inference_steps=n_steps)
    last_masks = story_pipeline.attention_store.last_mask

    dift_features = unet.latent_store.dift_features['201_0'][batch_size:]
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

    anchor_cache_first_stage.dift_cache = dift_features
    anchor_cache_first_stage.anchors_last_mask = last_masks

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(torch.device('cpu'))

    nn_map, nn_distances = cyclic_nn_map(dift_features, last_masks, LATENT_RESOLUTIONS, device)

    torch.cuda.empty_cache()
    gc.collect()

    # ------------------ #
    # Extended attention with nn_map #
    
    if perform_injection:
        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            anchors_cache=anchor_cache_second_stage,
                            num_inference_steps=n_steps)
        img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

        anchor_cache_second_stage.dift_cache = dift_features
        anchor_cache_second_stage.anchors_last_mask = last_masks

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(torch.device('cpu'))

        torch.cuda.empty_cache()
        gc.collect()
    else:
        img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    return out.images, img_all, anchor_cache_first_stage, anchor_cache_second_stage

def run_extra_generation(story_pipeline, prompts, concept_token, 
                         anchor_cache_first_stage, anchor_cache_second_stage,
                         seed=40, n_steps=50, mask_dropout=0.5,
                         same_latent=False, share_queries=True,
                         perform_sdsa=True, perform_injection=True,
                         downscale_rate=4, cache_cpu_offloading=False):
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = create_token_indices(prompts, batch_size, concept_token, tokenizer)

    default_attention_store_kwargs = {
        'token_indices': token_indices,
        'mask_dropout': mask_dropout
    }

    default_extended_attn_kwargs = {'extend_kv_unet_parts': ['up']}
    query_store_kwargs={'t_range': [0,n_steps//10], 'strength_start': 0.9, 'strength_end': 0.81836735}

    extra_batch_size = batch_size + 2
    if isinstance(seed, list):
        seed = [seed[0], seed[0], *seed]

    latents, g = create_latents(story_pipeline, seed, extra_batch_size, same_latent, device, float_type)
    latents = latents[2:]

    anchor_cache_first_stage.set_mode_inject()
    anchor_cache_second_stage.set_mode_inject()

    # ------------------ #
    # Extended attention First Run #

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(device)

    if perform_sdsa:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': [(1, n_steps)]}
    else:
        extended_attn_kwargs = {**default_extended_attn_kwargs, 't_range': []}

    print(extended_attn_kwargs['t_range'])
    out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                        attention_store_kwargs=default_attention_store_kwargs,
                        extended_attn_kwargs=extended_attn_kwargs,
                        share_queries=share_queries,
                        query_store_kwargs=query_store_kwargs,
                        anchors_cache=anchor_cache_first_stage,
                        num_inference_steps=n_steps)
    last_masks = story_pipeline.attention_store.last_mask

    dift_features = unet.latent_store.dift_features['261_0'][batch_size:]
    dift_features = torch.stack([gaussian_smooth(x, kernel_size=3, sigma=1) for x in dift_features], dim=0)

    anchor_dift_features = anchor_cache_first_stage.dift_cache
    anchor_last_masks = anchor_cache_first_stage.anchors_last_mask

    nn_map, nn_distances = anchor_nn_map(dift_features, anchor_dift_features, last_masks, anchor_last_masks, LATENT_RESOLUTIONS, device)

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(torch.device('cpu'))

    torch.cuda.empty_cache()
    gc.collect()

    # ------------------ #
    # Extended attention with nn_map #
    
    if perform_injection:

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(device)

        feature_injector = FeatureInjector(nn_map, nn_distances, last_masks, inject_range_alpha=[(n_steps//10, n_steps//3,0.8)], 
                                        swap_strategy='min', inject_unet_parts=['up', 'down'], dist_thr='dynamic')

        out = story_pipeline(prompt=prompts, generator=g, latents=latents, 
                            attention_store_kwargs=default_attention_store_kwargs,
                            extended_attn_kwargs=extended_attn_kwargs,
                            share_queries=share_queries,
                            query_store_kwargs=query_store_kwargs,
                            feature_injector=feature_injector,
                            anchors_cache=anchor_cache_second_stage,
                            num_inference_steps=n_steps)
        img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
        # display_attn_maps(story_pipeline.attention_store.last_mask, out.images)

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(torch.device('cpu'))

        torch.cuda.empty_cache()
        gc.collect()
    else:
        img_all = view_images([np.array(x) for x in out.images], display_image=False, downscale_rate=downscale_rate)
    
    return out.images, img_all

def transform_masks_to_images(masks, batch_size):
    return [Image.fromarray(mask.reshape(64, 64).cpu().numpy()) for mask in masks]