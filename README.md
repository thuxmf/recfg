# ReCFG &mdash; Official PyTorch implementation

> **Rectified Diffusion Guidance for Conditional Generation (CVPR 2025)** <br>
> Mengfei Xia, Nan Xue, Yujun Shen, Ran Yi, Tieliang Gong, Yong-Jin Liu <br>

[[Paper](https://arxiv.org/abs/2410.18737)]

Abstract: *Classifier-Free Guidance (CFG), which combines the conditional and unconditional score functions with two coefficients summing to one, serves as a practical technique for diffusion model sampling. Theoretically, however, denoising with CFG cannot be expressed as a reciprocal diffusion process, which may consequently leave some hidden risks during use. In this work, we revisit the theory behind CFG and rigorously confirm that the improper configuration of the combination coefficients (i.e., the widely used summing-to-one version) brings about expectation shift of the generative distribution. To rectify this issue, we propose ReCFG with a relaxation on the guidance coefficients such that denoising with ReCFG strictly aligns with the diffusion theory. We further show that our approach enjoys a closed-form solution given the guidance strength. That way, the rectified coefficients can be readily pre-computed via traversing the observed data, leaving the sampling speed barely affected. Empirical evidence on real-world data demonstrate the compatibility of our post-hoc design with existing state-of-the-art diffusion models, including both class-conditioned ones (e.g., EDM2 on ImageNet) and text-conditioned ones (e.g., SD3 on CC12M), without any retraining.*

## Installation

This repository is developed based on [EDM2](https://github.com/NVlabs/edm2), where you can find more detailed instructions on installation.

## Calculating lookup table

The preparation of lookup table consisting of expectation ratios could be directly achieved by `calculate_lookup.py`:

```shell
torchrun --standalone --nproc_per_node=8 calculate_lookup.py edm2 \
    --net=/path/to/net \
    --gnet=/path/to/gnet \
    --data=/path/to/data
```

This script will pre-compute the expectation ratios according to different NFEs, number of conditions, and number of traversals for each label. By default, `calculate_lookup.py` will use 100 conditions and 500 traversals per condition for NFE = 32, which coincides with the time cost of one single inference for 50,000 samples. Other conditions that are not involved will be represented via the mean of those involved ones. Please see more detailed analyses of **Section 4.4** in the paper.

To customize on your own models and samplers, it suffices to implement the sampler function, which compute **all** intermediate conditional and unconditional score functions at all timesteps, referring to `edm2_sampler` in `calculate_lookup.py`.

## Sampling with ReCFG

Sampling is implemented upon the original function in `generate_images.py`, with an additional argument `--coeffs` pre-computed by `calculate_lookup.py`, *i.e.*, simply use the script below:

```shell
torchrun --standalone --nproc_per_node=8 generate_images.py \
    --net=/path/to/net \
    --gnet=/path/to/gnet \
    --coeffs=/path/to/coeffs \
    --outdir=out
```

The averaged expectation ratio will be automatically applied to conditions that are not involved in lookup table preparation stage.

## References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{xia2025recfg,
  title={Rectified Diffusion Guidance for Conditional Generation},
  author={Xia, Mengfei and Xue, Nan and Shen, Yujun and Yi, Ran and Gong, Tieliang and Liu, Yong-Jin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```

## LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.
