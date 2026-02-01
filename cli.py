# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import datetime
import json
import os
import argparse
from run import load_pipeline, run_batch_generation, run_anchor_generation, run_extra_generation
import torch

def run_batch(gpu, float_type, seed=100, mask_dropout=0.5, same_latent=False,
              style="A photo of ", subject="a cute dog", concept_token=['dog'], prompts=None,
              out_dir = None, record_queries=False, invert=False, perform_feature_injection_bg_adain=False,
                perform_consistory_injection=False, run_experiments=None, attn_v_range=[3,10], attn_qk_range=[5,15],
                use_colab_folder_timestamp=True,
                perform_original_sdxl=False,
                perform_styled_injection=False,
                perform_ablations=False,
                perform_adain_ablation=False,
                perform_no_dift_ablation=False):
    
    print("Torch Cuda Available: ", torch.cuda.is_available())
    story_pipeline = load_pipeline(gpu, float_type)
    results = run_batch_generation(story_pipeline, 
                                    prompts, 
                                    concept_token, 
                                    seed,
                                    mask_dropout=mask_dropout, 
                                    same_latent=same_latent,
                                    record_queries=record_queries,
                                    invert=invert, 
                                    background_adain=None,
                                    perform_consistory_injection=perform_consistory_injection,
                                    attn_v_range=attn_v_range,
                                    attn_qk_range=attn_qk_range,
                                    use_colab_folder_timestamp=use_colab_folder_timestamp,
                                    perform_original_sdxl=perform_original_sdxl,
                                    perform_styled_injection=perform_styled_injection,
                                    perform_ablations=perform_ablations,
                                    perform_adain_ablation=perform_adain_ablation,
                                    perform_no_dift_ablation=perform_no_dift_ablation,
                                    custom_prompts=prompts,
                                    custom_concept_token=concept_token)

    for result in results:
        result.save(out_dir)

    return results

def run_cached_anchors(gpu, float_type, seed=40, mask_dropout=0.5, same_latent=False,
                style="A photo of ", subject="a cute dog", concept_token=['dog'],
                settings=["sitting in the beach", "standing in the snow"],
                cache_cpu_offloading=False, out_dir = None):
    
    story_pipeline = load_pipeline(gpu, float_type)
    prompts = [f'{style}{subject} {setting}' for setting in settings]
    anchor_prompts = prompts[:2]
    extra_prompts = prompts[2:]

    anchor_out_images, anchor_image_all, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(story_pipeline, anchor_prompts, concept_token, 
                                                                                                        seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                                                                                                        cache_cpu_offloading=cache_cpu_offloading)
    
    if out_dir is not None:
        for i, image in enumerate(anchor_out_images):
            image.save(f'{out_dir}/anchor_image_{i}.png')

    for i, extra_prompt in enumerate(extra_prompts):
        extra_out_images, extra_image_all = run_extra_generation(story_pipeline, [extra_prompt], concept_token, anchor_cache_first_stage, anchor_cache_second_stage, 
                                                    seed=seed, mask_dropout=mask_dropout, same_latent=same_latent, cache_cpu_offloading=cache_cpu_offloading)
        
        if out_dir is not None:
            extra_out_images[0].save(f'{out_dir}/extra_image_{i}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', default="batch", type=str, required=False) # batch, cached

    parser.add_argument('--gpu', default=0, type=int, required=False)
    parser.add_argument('--float_type', default=16, type=int, required=False)
    parser.add_argument('--seed', default=40, type=int, required=False)
    parser.add_argument('--mask_dropout', default=0.5, type=float, required=False)
    parser.add_argument('--same_latent', default=False, type=bool, required=False)

    parser.add_argument('--style', default="A photo of ", type=str, required=False)
    parser.add_argument('--subject', default="a cute dog", type=str, required=False)
    parser.add_argument('--concept_token', default=["dog"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--prompts', default=["portrait of a happy girl wearing headphones, anime drawing", 
                                              "portrait of a happy girl having a picnic, realistic photo",
                                              "portrait a happy girl in the snow, black and white sketch"], 
                        type=str, nargs='*', required=False)
    parser.add_argument('--cache_cpu_offloading', default=False, type=bool, required=False)
    parser.add_argument('--perform_feature_injection_bg_adain', default=False, type=bool, required=False)
    parser.add_argument('--perform_consistory_injection', default=False, type=bool, required=False)
    parser.add_argument('--attn_v_range', default=[3,10], type=int, nargs='*', required=False)
    parser.add_argument('--attn_qk_range', default=[5,15], type=int, nargs='*', required=False)

    parser.add_argument('--record_queries', default=False, type=bool, required=False)
    parser.add_argument('--invert', default=False, type=bool, required=False)
    
    parser.add_argument('--subfolder', default=None, type=str, required=False)
    parser.add_argument('--use_colab_folder_timestamp', default=None, type=bool, required=False)
    
    # Experiment Options
    parser.add_argument('--perform_original_sdxl', default=False, type=bool, required=False)
    parser.add_argument('--perform_styled_injection', default=False, type=bool, required=False)
    parser.add_argument('--perform_ablations', default=False, type=bool, required=False)
    parser.add_argument('--perform_adain_ablation', default=False, type=bool, required=False)
    parser.add_argument('--perform_no_dift_ablation', default=False, type=bool, required=False)
    
    parser.add_argument('--out_dir', default=None, type=str, required=False)
    parser.add_argument('--perform_background_adain', default=True, type=bool, required=False)

    args = parser.parse_args()
    
    if args.float_type == 16:
        float_type = torch.float16
    else:
        float_type = torch.float32

    for concept in args.concept_token:
        if concept not in args.subject:
            print("Concept token should be part of the subject")


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.out_dir += f"/{'-'.join(args.concept_token)}_{timestamp}_seed{args.seed}"
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    
    metadata_path = os.path.join(args.out_dir, f"metadata.json")
    with open(metadata_path, mode="w", newline="") as file:
        json.dump(vars(args), file, indent=4)

    if args.run_type == "batch":
        run_batch(args.gpu, float_type, args.seed, args.mask_dropout, args.same_latent, args.style, 
                  args.subject, args.concept_token, args.prompts, args.out_dir,args.record_queries,
                  args.invert, args.perform_feature_injection_bg_adain, args.perform_consistory_injection,
                    args.run_experiments, args.attn_v_range, args.attn_qk_range,
                    subfolder=args.subfolder,
                    use_colab_folder_timestamp=args.use_colab_folder_timestamp,
                    perform_original_sdxl=args.perform_original_sdxl,
                    perform_styled_injection=args.perform_styled_injection,
                    perform_ablations=args.perform_ablations,
                    perform_adain_ablation=args.perform_adain_ablation,
                    perform_no_dift_ablation=args.perform_no_dift_ablation
                )
    elif args.run_type == "cached":
        run_cached_anchors(args.gpu, float_type, args.seed, args.mask_dropout, args.same_latent, args.style, 
                           args.subject, args.concept_token, args.settings, args.cache_cpu_offloading, args.out_dir)
    else:
        print("Invalid run type")