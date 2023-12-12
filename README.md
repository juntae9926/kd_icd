

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
- RegNet-Y
- MobileViT

## Performances

<table style="margin: auto">
  <tr>
    <th>model</th>
    <th># of<br />params[M]</th>
    <th>Feature Size</th>
    <th>µAP</th>
    <th>Acc@1</th>
    <th>download</th>
  </tr>
  <tr>
    <td>EfficientNet-b0</td>
    <td align="right">4.7</td>
    <td align="right">512</td>
    <td align="right">67.4%</td>
    <td align="right">75.0%</td>
    <td><a href="https://drive.google.com/file/d/10Jxr4aCiBA5nkmU3kBJaopzOrSvqi9RF/view?usp=drive_link">link</a></td>
  </tr>
  <tr>
    <td>MobileNet-v3</td>
    <td align="right">4.9</td>
    <td align="right">512</td>
    <td align="right">68.0%</td>
    <td align="right">74.3%</td>
    <td><a href="https://drive.google.com/file/d/1Tn0iLFGfePdpNHgK2h2t8PrQz-Bje1fY/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>MobileViT-s</td>
    <td align="right">5.3</td>
    <td align="right">512</td>
    <td align="right">69.2%</td>
    <td align="right">75.4%</td>
    <td><a href="https://drive.google.com/file/d/1Snnt5TLIZudDesYHTMIcSvmzUbzQemvg/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>Regnet_y</td>
    <td align="right">6.0</td>
    <td align="right">512</td>
    <td align="right">70.9%</td>
    <td align="right">76.5%</td>
    <td><a href="https://drive.google.com/file/d/1qgy2VyJyu_tLIrZ2dA1LA1g1wGvW-DVt/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td align="right">11.2</td>
    <td align="right">512</td>
    <td align="right">62.5%</td>
    <td align="right">69.5%</td>
    <td><a href="https://drive.google.com/file/d/11bb5R225iRCj74yPWXfM6o9BVitmRDmp/view?usp=sharing">link</a></td>
  </tr>
</table>


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

### Training
```
sh train.sh
```

### Evaluation
```
sh disc_eval.sh
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
