# Deep Online Video Stabilization
https://arxiv.org/pdf/1802.08091.pdf

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- tensorflow-gpu==1.3.0
- numpy
- ...

## Getting Started
### Installation
download demo.zip
```bash
unzip demo.zip
mv demo/models deep-online-video-stabilization-deploy/
mv demo/datas deep-online-video-stabilization-deploy/
cd deep-online-video-stabilization-deploy
mkdir output
```

### Testing
```bash
python3 -u deploy_bundle.py --model-dir ./models/v2_93/ --model-name model-90000 --before-ch 31 --deploy-vis --gpu_memory_fraction 0.9 --output-dir ./output/v2_93/Regular  --test-list datas/Regular/list.txt --prefix datas/Regular;
```

