# ReCFG &mdash; Official PyTorch implementation

> **Rectified Diffusion Guidance for Conditional Generation (CVPR 2025)** <br>
> Mengfei Xia, Nan Xue, Yujun Shen, Ran Yi, Tieliang Gong, Yong-Jin Liu <br>

[[Paper](https://arxiv.org/abs/2410.18737)]

Abstract: *Classifier-Free Guidance (CFG), which combines the conditional and unconditional score functions with two coefficients summing to one, serves as a practical technique for diffusion model sampling. Theoretically, however, denoising with CFG cannot be expressed as a reciprocal diffusion process, which may consequently leave some hidden risks during use. In this work, we revisit the theory behind CFG and rigorously confirm that the improper configuration of the combination coefficients (i.e., the widely used summing-to-one version) brings about expectation shift of the generative distribution. To rectify this issue, we propose ReCFG with a relaxation on the guidance coefficients such that denoising with ReCFG strictly aligns with the diffusion theory. We further show that our approach enjoys a closed-form solution given the guidance strength. That way, the rectified coefficients can be readily pre-computed via traversing the observed data, leaving the sampling speed barely affected. Empirical evidence on real-world data demonstrate the compatibility of our post-hoc design with existing state-of-the-art diffusion models, including both class-conditioned ones (e.g., EDM2 on ImageNet) and text-conditioned ones (e.g., SD3 on CC12M), without any retraining.*

## TODO List

- [ ] Release code for pre-computing rectified coefficients.
- [ ] Release inference code using ReCFG.

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
