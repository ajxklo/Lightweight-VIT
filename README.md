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
   - **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**
      [[GitHub](https://github.com/whai362/PVT)] [[PDF](https://arxiv.org/abs/2102.12122)]
   - **Vision Transformers with Hierarchical Attention**
       [[GitHub](https://github.com/yun-liu/HAT-Net)] [[PDF](https://arxiv.org/abs/2106.03180)]
   - **Twins: Revisiting the Design of Spatial Attention in Vision Transformers**
       [[GitHub](https://github.com/Meituan-AutoML/Twins)] [[PDF](https://arxiv.org/abs/2104.13840)]
   - **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
       [[GitHub](https://github.com/microsoft/Swin-Transformer)] [[PDF](https://arxiv.org/abs/2103.14030)]
   - **MaxViT: Multi-Axis Vision Transformer**
       [[GitHub](https://github.com/google-research/maxvit)] [[PDF](https://arxiv.org/abs/2204.01697)]
   - **FDViT: Improve the Hierarchical Architecture of Vision Transformer**
       [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_FDViT_Improve_the_Hierarchical_Architecture_of_Vision_Transformer_ICCV_2023_paper.pdf)]
   - **HiViT: Hierarchical Vision Transformer Meets Masked Image Modeling**
       [[PDF](https://arxiv.org/abs/2205.14949)]
   -  **Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles**
       [[GitHub](https://github.com/facebookresearch/hiera)] [[PDF](https://arxiv.org/abs/2306.00989)]
   - **HIRI-ViT: Scaling Vision Transformer with High Resolution Inputs**
       [[PDF](https://arxiv.org/abs/2403.11999)]
## Dynamic Network
1. ### Dynamic Resolution
   - **Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer**
       [[GitHub](https://github.com/YifanXu74/Evo-ViT)] [[PDF](https://arxiv.org/pdf/2108.01390)]
   - **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification**
       [[GitHub](https://github.com/raoyongming/DynamicViT)] [[PDF](https://arxiv.org/abs/2106.02034#)]
   - **EViT: An Eagle Vision Transformer with Bi-Fovea Self-Attention**
       [[GitHub](https://github.com/nkusyl/EViT)] [[PDF](https://arxiv.org/abs/2310.06629)]
   - **SPViT: Enabling Faster Vision Transformers via Soft Token Pruning**
       [[PDF](https://arxiv.org/abs/2112.13890)]
   - **HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers**
       [[PDF](https://arxiv.org/abs/2211.08110)]
   - **HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers**
       [[PDF](https://arxiv.org/abs/2211.08110)]
   - **CF-ViT: A General Coarse-to-Fine Method for Vision Transformer**
       [[PDF](https://arxiv.org/abs/2203.03821)]
   - **No Token Left Behind: Efficient Vision Transformer via Dynamic Token Idling**
       [[PDF](https://arxiv.org/abs/2310.05654v2)]
   - **ATS: Adaptive Token Sampling For Efficient Vision Transformers**
       [[PDF](https://arxiv.org/abs/2111.15667v1)]
   - **TPC-ViT: Token Propagation Controller for Efficient Vision Transformer**
       [[PDF](https://arxiv.org/abs/2401.01470)]
   - **Multi-Scale And Token Mergence: Make Your ViT More Efficient**
       [[PDF](https://arxiv.org/abs/2306.04897)]
   - **Super Vision Transformer**
       [[GitHub](https://github.com/lmbxmu/SuperViT)] [[PDF](https://arxiv.org/pdf/2205.11397v2)]
   - **Token Merging: Your ViT But Faster**
       [[GitHub](https://github.com/facebookresearch/ToMe)] [[PDF](https://arxiv.org/abs/2210.09461)]
   - **Token Fusion: Bridging the Gap between Token Pruning and Token Merging**
       [[PDF](https://arxiv.org/pdf/2312.01026)]
   - **Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer**
       [[GitHub](https://github.com/zengwang430521/TCFormer.git)] [[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_Not_All_Tokens_Are_Equal_Human-Centric_Visual_Analysis_via_Token_CVPR_2022_paper.pdf)]
2. ### Depth Adaptation
   - **A-ViT: Adaptive Tokens for Efficient Vision Transformer**
       [[GitHub](https://github.com/NVlabs/A-ViT)] [[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf)]
   - **Distillation-Based Training for Multi-Exit Architectures**
       [[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.pdf)]
   - **Single-Layer Vision Transformers for More Accurate Early Exits with Less Overhead**
       [[PDF](https://arxiv.org/abs/2105.09121)]
   - **Multi-Exit Vision Transformer for Dynamic Inference**
       [[PDF](https://arxiv.org/abs/2106.15183v1)]
   - **LGViT: Dynamic Early Exiting for Accelerating Vision Transformer**
       [[PDF](https://arxiv.org/abs/2308.00255)]
   - **Dyn-Adapter: Towards Disentangled Representation for Efficient Visual Recognition**
       [[PDF](https://arxiv.org/abs/2407.14302)]
   - **CF-ViT: A General Coarse-to-Fine Method for Vision Transformer**
       [[GitHub](https://github.com/ChenMnZ/CF-ViT)] [[PDF](https://arxiv.org/abs/2203.03821)]
   - **AdaViT: Adaptive Vision Transformers for Efficient Image Recognition**
       [[PDF](https://arxiv.org/pdf/2111.15668)]
## Knowledge Distillation
1. ### Feature Knowledge Distillation
2. ### Response Knowledge Distillation
