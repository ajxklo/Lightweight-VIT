# Lightweight-VIT
## Abstract
The transformer architecture has achieved significant success in natural language processing, motivating its adaptation to computer vision tasks. 
Unlike convolutional neural networks, vision transformers inherently capture long-range dependencies and enable parallel processing, yet lack inductive biases and efficiency benefits, facing significant computational and memory challenges that limit its real-world applicability. 
This paper surveys various online strategies for generating lightweight vision transformers for image recognition, focusing on three key areas: efficient component design, dynamic network, and knowledge distillation. 
We evaluate the corresponding exploration for each topic using ImageNet benchmark, analyzing trade-offs among precision, parameters, throughput, and more to highlight their respective advantages, disadvantages, and flexibility. 
Finally, we propose future research directions and potential   challenges in the lightweighting of vision transformers, with the aim of inspiring further exploration and providing practical guidance for the community.
## Overview of PaPer
![overview](picture/overview.png)
## content
- [Efficient ViT Components](#efficient-vit-components)
  - [Embedding Structure Design](embedding-structure-design)
  - [Efficient Position Encoding](efficient-position-encoding)
  - [Efficient Token Update](efficient-token-update)
  - [Framework Design](framework-design)
- [Dynamic Network](#dynamic-network)
  - [Dynamic Resolution](dynamic-resolution)
  - [Depth Adaptation](depth-adaptation)
- [Knowledge Distillation](#knowledge-distillation)
  - [Feature Knowledge Distillation](feature-knowledge-distillation)
  - [Response Knowledge Distillation](response-knowledge-distillation)


## Efficient ViT Components
1. ### Embedding Structure Design
    - **CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification**
     [[GitHub](https://github.com/IBM/CrossViT)] [[PDF](https://arxiv.org/abs/2103.14899)]
    - **CvT: Introducing Convolutions to Vision Transformers**
     [[GitHub](https://github.com/leoxiaobin/CvT)] [[PDF](https://arxiv.org/pdf/2103.15808)]
    - **MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer**
     [[GitHub](https://github.com/apple/ml-cvnets)] [[PDF](https://arxiv.org/abs/2103.15808)]
    - **Mobile-Former: Bridging MobileNet and Transformer**
     [[PDF](https://arxiv.org/abs/2108.05895)]
    - **Shunted Self-Attention via Multi-Scale Token Aggregation**
     [[GitHub](https://github.com/OliverRensu/Shunted-Transformer)] [[PDF](https://arxiv.org/abs/2111.15193)]
    - **Patch Slimming for Efficient Vision Transformers**
     [[PDF](https://arxiv.org/abs/2106.02852)]
    - **Shunted Self-Attention via Multi-Scale Token Aggregation**
      [[GitHub](https://github.com/OliverRensu/Shunted-Transformer)] [[PDF](https://arxiv.org/abs/2111.15193)]
    - **RepViT: Revisiting Mobile CNN From ViT Perspective**
      [[GitHub](https://github.com/THU-MIG/RepViT)] [[PDF](https://arxiv.org/abs/2307.09283)]
    - **TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation**
      [[GitHub](https://github.com/hustvl/TopFormer)] [[PDF](https://arxiv.org/abs/2204.05525v1)]
2. ### Efficient Position Encoding
   - **Rotary Position Embedding for Vision Transformer**
     [[GitHub](https://github.com/naver-ai/rope-vit)] [[PDF](https://arxiv.org/abs/2403.13298#:~:text=Rotary%20Position%20Embedding%20%28RoPE%29%20performs%20remarkably%20on%20language,in%20a%20way%20similar%20to%20the%20language%20domain.)]
    - **Parameterization of Cross-Token Relations with Relative Positional Encoding for Vision MLP**
      [[GitHub](https://github.com/Zhicaiwww/PosMLP)] [[PDF](https://arxiv.org/abs/2207.07284)]
    - **Lightweight Structure-Aware Attention for Visual Understanding**
      [[PDF](https://arxiv.org/abs/2211.16289v1)]
    - **Functional Interpolation for Relative Positions Improves Long Context Transformers**
      [[PDF](https://arxiv.org/abs/2310.04418)]
    - **RELATIVE POSITIONAL ENCODING FAMILY VIA UNITARY TRANSFORMATION**
      [[PDF](https://openreview.net/pdf?id=xMWFqb5Uyk)]
    - **Conditional Positional Encodings for Vision Transformers**
      [[GitHub](https://github.com/Meituan-AutoML/CPVT)] [[PDF](https://arxiv.org/abs/2102.10882)]
3. ### Efficient Token Update
    - **Global Filter Networks for Image Classification**
      [[GitHub](https://github.com/raoyongming/GFNet)] [[PDF](https://arxiv.org/abs/2107.00645)]
    - **Focal Self-attention for Local-Global Interactions in Vision Transformers**
      [[PDF](https://arxiv.org/abs/2107.00641)]
    - **Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet**
      [[GitHub](https://github.com/yitu-opensource/T2T-ViT)] [[PDF](https://arxiv.org/abs/2101.11986)]
    - **CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows**
      [[GitHub](https://github.com/microsoft/CSWin-Transformer)] [[PDF](https://arxiv.org/abs/2107.00652)]
    - **MaxViT: Multi-Axis Vision Transformer**
      [[GitHub](https://github.com/google-research/maxvit)] [[PDF](https://arxiv.org/abs/2204.01697)]
    - **Skip-Attention: Improving Vision Transformers by Paying Less Attention**
      [[PDF](https://arxiv.org/abs/2301.02240)]
    - **SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications**
      [[GitHub](https://github.com/Amshaker/SwiftFormer)] [[PDF](https://arxiv.org/abs/2303.15446)]
    - **BiFormer: Vision Transformer with Bi-Level Routing Attention**
      [[GitHub](https://github.com/rayleizhu/BiFormer)] [[PDF](https://arxiv.org/abs/2303.08810)]
    - **ReViT: Enhancing Vision Transformers Feature Diversity with Attention Residual Connections**
      [[PDF](https://arxiv.org/abs/2402.11301)]
    - **Vision Transformer with Sparse Scan Prior**
      [[GitHub](https://github.com/qhfan/SSViT)] [[PDF](https://arxiv.org/abs/2405.13335)]
    - **ConvMLP: Hierarchical Convolutional MLPs for Vision**
      [[GitHub](https://github.com/SHI-Labs/Convolutional-MLPs)] [[PDF](https://ar5iv.labs.arxiv.org/html/2109.04454)]
    - **CycleMLP: A MLP-like Architecture for Dense Prediction**
      [[GitHub](https://github.com/ShoufaChen/CycleMLP)] [[PDF](https://arxiv.org/pdf/2107.10224v1)]
    - **ResMLP: Feedforward networks for image classification with data-efficient training**
      [[PDF](https://arxiv.org/abs/2105.03404)]
    - **Hire-MLP: Vision MLP via Hierarchical Rearrangement**
      [[GitHub](https://github.com/ggjy/Hire-Wave-MLP.pytorch)] [[PDF](https://arxiv.org/abs/2108.13341)]
4. ### Framework Design
## Dynamic Network
1. ### Dynamic Resolution
2. ### Depth Adaptation
## Knowledge Distillation
1. ### Feature Knowledge Distillation
2. ### Response Knowledge Distillation
