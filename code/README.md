# Domain Adaptation and Generalization for Medical Image Analysis

## Experimental Requirements
Some important required packages include:
* [Pytorch][torch_link] version >= 1.8.0.
* TensorboardX
* Python == 3.8 
* Some basic python packages such as Numpy.

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/


## 1. Contrastive Semi-supervised Learning for Domain Adaptive Segmentation Across similar Anatomical Structures
This repository provides the official code for "[Contrastive Semi-supervised Learning for Domain Adaptive Segmentation Across similar Anatomical Structures][paper_link]".

[paper_link]:https://arxiv.org/abs/2208.08605

<center><img src='./picture/cscada/model.jpg', width='72%'></center>
<center>Fig. 1. Flowchart of CS-CADA.</center>

### Usages
#### For circular structure segmentation
1. First, you should download the retinal dataset at [REFUGE Challenge][data1_link]. We only used the 360 non-glaucoma images in this dataset and central-cropped and resized the images to 256. Second, you should download the CMR dataset at [MS-CMRSeg 2019][data2_link]. We only used the bSSFP-sequence dataset and splited the 3D volumes into slice, and also central cropped and resized to 256.

[data1_link]:https://refuge.grand-challenge.org/
[data2_link]:http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg19/index.html


2. To train CS-CADA in circular structure segmentation, run:
```
python train_cscada.py
```

3. To evaluate the trained model in CMR images for Left Ventricle (LV) and left ventricular Myocardium (Myo) segmentaiton, run:
```
python test_mscmrseg.py
```
### Citation
If this project is helpful for your research, please cite the following works:
```
@article{gu2022contrastive,
  title={Contrastive Semi-supervised Learning for Domain Adaptive Segmentation Across Similar Anatomical Structures},
  author={Gu, Ran and Zhang, Jingyang and Wang, Guotai and Lei, Wenhui and Song, Tao and Zhang, Xiaofan and Li, Kang and Zhang, Shaoting},
  journal={arXiv preprint arXiv:2208.08605},
  year={2022}
}
@inproceedings{zhang2021ss,
  title={SS-CADA: A semi-supervised cross-anatomy domain adaptation for coronary artery segmentation},
  author={Zhang, Jingyang and Gu, Ran and Wang, Guotai and Xie, Hongzhi and Gu, Lixu},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={1227--1231},
  year={2021},
  organization={IEEE}
}
```

### Acknowledgement
Part of the code is revised from [UA-MT][uamt].

[uamt]:https://github.com/yulequan/UA-MT
