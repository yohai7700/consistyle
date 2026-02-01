# ConsiStyle: Style Diversity in Training-Free Consistent T2I Generation [SIGGRAPH Asia 2026, ACM ToG]

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2505.20626)

### [[Project Website](https://jbruner23.github.io/consistyle/)]

> **ConsiStory: Training-Free Consistent Text-to-Image Generation**<br>
> Yohai Mazuz<sup>1*</sup>, Janna Bruner<sup>1*</sup>, Lior Wolf<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>Amazon<br>
> <sup>\*</sup>Equal contribution

![](https://raw.githubusercontent.com/jbruner23/consistyle/main/static/images/cover_v6.png)

> **Abstract**: <br>
> In text-to-image models, consistent character generation is the task of achieving text alignment while maintaining the subject's appearance across different prompts. However, since style and appearance are often entangled, the existing methods struggle to preserve consistent subject characteristics while adhering to varying style prompts. Current approaches for consistent text-to-image generation typically rely on large-scale fine-tuning on curated image sets or per-subject optimization, which either fail to generalize across prompts or do not align well with textual descriptions. Meanwhile, training-free methods often fail to maintain subject consistency across different styles. In this work, we introduce a training-free method that, for the first time, jointly achieves style preservation and subject consistency across varied styles. The attention matrices are manipulated such that Queries and Keys are obtained from the anchor image(s) that are used to define the subject, while the Values are imported from a parallel copy that is not subject-anchored. Additionally, cross-image components are added to the self-attention mechanism by expanding the Key and Value matrices. To do without shifting from the target style, we align the statistics of the Value matrices. As is demonstrated in a comprehensive battery of qualitative and quantitative experiments, our method effectively decouples style from subject appearance and enables faithful generation of text-aligned images with consistent characters across diverse styles.

## Setup

```bash
conda env create --file environment.yml
```

## Nvidia License

This repository contains code from [ConsiStory](https://github.com/NVlabs/consistory) by NVIDIA.
NVIDIA's modifications are subject to the [NVIDIA Source Code License-NC](NVIDIA_LICENSE)
(non-commercial use only).

## Citation

If you make use of our work, please cite our paper:

```
@article{mazuz2025consistyle,
author    = {Yohai Mazuz and Janna Bruner and Lior Wolf},
title     = {ConsiStyle: Style Diversity in Training-Free Consistent T2I Generation},
journal   = {ACM Transactions on Graphics (TOG)},
year      = {2025},
volume    = {44},
number    = {6},
article   = {263},
month     = {12},
doi       = {10.1145/3763303}
}
```
