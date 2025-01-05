# Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering (NeuIPS 2024)

Meng Wei<sup>1</sup>, [Qianyi Wu<sup>1</sup>](), [Jianmin Zheng<sup>2</sup>](https://personal.ntu.edu.sg/asjmzheng/), [Hamid Rezatofighi<sup>1</sup>](https://research.monash.edu/en/persons/hamid-rezatofighi), [Jianfei Cai<sup>1*</sup>](https://jianfei-cai.github.io/) <br />
<sup>1</sup>Monash University <sup>2</sup>Nanyang Technological University  <sup>*</sup>Corresponding Author

[[`arxiv`](https://arxiv.org/abs/2410.20593)]
<!-- [[`Project Page`]()] -->

## News
**[2025.01.05]** Code released.

## TODO List
- [ ] Improve the rasterizer.
- [ ] Clean-up codes.

## Environmnent Setups
We tested our codes on RTX 3090 with Ubuntu 22.04, and cuda 12.2.

1. Clone this repo:
```
git clone https://github.com/Meng-Wei/Normal-GS.git --recursive
cd Normal-GS
```

2. Install Packages
```
conda env create --file environmnet.yml
conda activate normal_gs
```

3. Data preparation: please follow [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)

4. Training and Evaluation commands
```
python train.py --eval -s data/mipnerf360/bonsai --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 30_000 --ref --idiv

python train.py --eval -s data/tandt/truck --lod 0 --gpu -1 --voxel_size 0.01 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 30_000 --ref --idiv

python train.py --eval -s data/blending/drjohnson --lod 0 --gpu -1 --voxel_size 0.005 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 30_000 --ref --idiv
```

## Citation

```bibtex
@inproceedings{wei2024normalgs,
  title={Normal-GS: 3D Gaussian Splatting with Normal-Involved Rendering},
  author={Wei, Meng and Wu, Qianyi and Zheng, Jianmin and Rezatofighi, Hamid and Cai, Jianfei},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## LICENSE

Please see the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

We thank all authors from [Ref-NeRF](https://github.com/google-research/multinerf), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS), [GaussianShader](https://github.com/Asparagus15/GaussianShader), and [Gaussian Surfels](https://github.com/turandai/gaussian_surfels) for sharing their codes and presenting excellent work.

Our codes are based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Ref-NeRF](https://github.com/google-research/multinerf), and [Scaffold-GS](https://github.com/city-super/Scaffold-GS).

```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@misc{multinerf2022,
      title={{MultiNeRF}: {A} {Code} {Release} for {Mip-NeRF} 360, {Ref-NeRF}, and {RawNeRF}},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
      url={https://github.com/google-research/multinerf},
}

@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}
```