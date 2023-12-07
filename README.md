

# Self-supervised Knowledge Distillation using Dynamic Memory Bank for Image Copy Detection.
Implementation of "Self-supervised Knowledge Distillation using Dynamic Memory Bank for Image Copy Detection" that is Juntae's Masters Thesis 

This repository contains strong SSCD baseline implementation.
"[A Self-Supervised Descriptor for Image Copy Detection](https://cvpr2022.thecvf.com/)[CVPR 2022]".

## About this codebase

This implementation is built on [Pytorch Lightning](https://pytorchlightning.ai/),
with some components from [Classy Vision](https://classyvision.ai/).

## Datasets used
- DISC (Facebook Image Similarity Challenge 2021)
- Copydays

## Unsupervised Knowledge Distillation
- similarity distillation 
- KoLeo regularization
- DirectCLR contrastive learning

## Teacher Models(Pretrained)
- SSCD: ResNet-50
- DINO: ViT-B/16

## Student Models
- ResNet-18
- EfficientNet-B0
- MobileNet-V3-Large
- ViT-S
- MobileViT

## How to use

### Install miniconda3
```
wget \
https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
&& mkdir /root/.conda \
&& bash Miniconda3-latest-Linux-x86_64.sh -b \
&& rm -f Miniconda3-latest-Linux-x86_64.sh

.bashrc에 다음 코드 추가

export PATH=”/root/miniconda3/bin:$PATH”

source .bashrc
```

### 컨테이너 재시작

```
exit → attach

conda create -n test python=3.8
```

### Library 설치
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r reqirements.txt
conda install -c pytorch faiss-gpu
```

## Citation
```
@article{juntae2023,
  title={Self-supervised Knowledge Distillation using Dynamic Memory Bank for Image Copy Detection},
  author={Juntae Kim},
  journal={Master's Thesis},
  year={2023}
}

@article{pizzi2022self,
  title={A Self-Supervised Descriptor for Image Copy Detection},
  author={Pizzi, Ed and Roy, Sreya Dutta and Ravindra, Sugosh Nagavara and Goyal, Priya and Douze, Matthijs},
  journal={Proc. CVPR},
  year={2022}
}
```
