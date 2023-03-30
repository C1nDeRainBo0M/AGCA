# AGCA: An Adaptive Graph Channel Attention Module for Steel Surface Defect Detection
This work has been accepted for publication in the IEEE Transactions on Instrumentation and Measurement (https://ieeexplore.ieee.org/document/10050536).

***AGCA.py*** is the pytorch code implementation of the AGCA attention module.


## Abstrct
Surface defect detection is an important part of the steel production process. Recently, attention mechanisms have been widely used in steel surface defect detection to ensure product quality. Existing attention modules cannot distinguish the difference between steel surface images and natural images. Therefore, we propose an Adaptive Graph Channel Attention (AGCA) module which introduces graph convolutional theory into channel attention. The AGCA module takes each channel as a feature vertex, and their relationship is represented by an adjacency matrix. We perform non-local operations on features by analyzing graphs constructed in AGCA. The operation significantly improves the feature representation capability. Like other attention modules, AGCA has lightweight and plug-and-play characteristics. It enables the module easily embedded into defect detection networks. The experimental results on various backbone networks and datasets show that AGCA outperforms state-of-the-art methods.


<div align='center'>
<img src = 'AGCA.png'>
</div>

## Citation
If you found the study useful for you, please consider citing it.
```
@ARTICLE{Xiang2023AGCA,  
author  = {Xiang, Xin and Wang, Zenghui and Zhang, Jun and Xia, Yi and Chen, Peng and Wang, Bing},  
journal = {IEEE Transactions on Instrumentation and Measurement},   
title   = {AGCA: An Adaptive Graph Channel Attention Module for Steel Surface Defect Detection},   
year    = {2023},  
volume  = {72},  
number  = {},  
pages   = {1-12},  
doi     = {10.1109/TIM.2023.3248111}}
```
