# Deep Online Video Stabilization with Multi-Grid Warping Transformation Learning
https://ieeexplore.ieee.org/document/8554287

## Other Versions
You can get the training code and data in this project: https://github.com/cxjyxxme/deep-online-video-stabilization

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- tensorflow-gpu==1.3.0 (Latest compatible version==1.15 [with pip](https://www.tensorflow.org/install/pip))
- numpy
- ...

## Getting Started
### Installation
download demo.zip at https://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/demo.zip
```bash
unzip demo.zip
mv demo/models deep-online-video-stabilization-deploy/
mv demo/datas deep-online-video-stabilization-deploy/
cd deep-online-video-stabilization-deploy
mkdir output
```

### Testing
With reference ground truth stable available in `datas/Regular`-
```bash
python3 -u deploy_bundle.py --model-dir ./models/v2_93/ --model-name model-90000 --before-ch 31 --deploy-vis --gpu_memory_fraction 0.9 --output-dir ./output/v2_93/Regular  --test-list datas/Regular/list.txt --prefix datas/Regular;
```

With no reference ground truth stable available-
```bash
python3 -u deploy_no_stable.py --model-dir ./models/v2_93/ --model-name model-90000 --gpu_memory_fraction 0.9 --output-dir ./output/v2_93/Regular  --test-list datas/Regular/list.txt --prefix datas/Regular;
```
### Dataset
DeepStab dataset (7.9GB)
http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip

## Citation

If you find this useful for your research, please cite the following paper.

```
@ARTICLE{StabNet, 
author={M. Wang and G. Yang and J. Lin and S. Zhang and A. Shamir and S. Lu and S. Hu}, 
journal={IEEE Transactions on Image Processing}, 
title={Deep Online Video Stabilization with Multi-Grid Warping Transformation Learning}, 
year={2018}, 
volume={}, 
number={}, 
pages={1-1}, 
}
```
