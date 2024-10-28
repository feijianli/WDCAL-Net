# WDCAL-Net
## Introduction
- WDCAL-Net is an open source, PyTorch-based segmentation framework for 3D medical image. 

## Installation
#### Environment
- Ubuntu 16.04.12
- Python 3.6+
- Pytorch 1.10.1
- CUDA 11.6

1.Git clone
```
git clone https://github.com/feijianli/WDCAL-Net.git
```

2.Install Nvidia Apex
- Perform the following command:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

3.Install dependencies
```
pip install -r requirements.txt
```

## Get Started
### preprocessing
1. Download [Synapse](https://www.synapse.org/Synapse:syn3193805/wiki/217789), resulting in 19 training images and masks, 5 validation images.
2. Copy image and mask to 'FlareSeg/dataset/' folder.
3. Edit the 'FlareSeg/data_prepare/config.yaml'. 
   'DATA_BASE_DIR'(Default: FlareSeg/dataset/) is the base dir of databases.
   If set the 'IS_SPLIT_5FOLD'(Default: False) to true, 5-fold cross-validation datasets will be generated.
4. Run the data preprocess with the following command:
```bash
cd FlareSeg/data_prepare
python run.py
```
The image data and lmdb file are stored in the following structure:
```wiki
DATA_BASE_DIR directory structure：
├── train_images
   ├── train_000_0000.nii.gz
   ├── train_001_0000.nii.gz
   ├── train_002_0000.nii.gz
   ├── ...
├── train_mask
   ├── train_000.nii.gz
   ├── train_001.nii.gz
   ├── train_002.nii.gz
   ├── ...
└── val_images
    ├── validation_001_0000.nii.gz
    ├── validation_002_0000.nii.gz
    ├── validation_003_0000.nii.gz
    ├── ...
├── file_list
    ├──'train_series_uids.txt', 
    ├──'val_series_uids.txt',
    ├──'lesion_case.txt',
├── db
    ├──seg_raw_train        
    ├──seg_raw_test         
    ├──seg_train_database  
    ├──seg_val_database      
    ├──seg_pre-process_database # Temporary database.
    ├──seg_train_fold_1
    ├──seg_val_fold_1
├── coarse_image
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
├── coarse_mask
    ├──160_160_160
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
├── fine_image
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├──  ...
├── fine_mask
    ├──192_192_192
          ├── train_000.npy
          ├── train_001.npy
          ├── ...
```
The data information is stored in the lmdb file with the following format:
```wiki
{
    series_id = {
        'image_path': data.image_path,
        'mask_path': data.mask_path,
        'smooth_mask_path': data.smooth_mask_path,
        'coarse_image_path': data.coarse_image_path,
        'coarse_mask_path': data.coarse_mask_path,
        'fine_image_path': data.fine_image_path,
        'fine_mask_path': data.fine_mask_path
    }
}
```

### Training
Remark: Coarse segmentation is trained on Nvidia GeForce 3090(Number:1), while fine segmentation on Nvidia A6000(Number:1). If you use different hardware, please set the "ENVIRONMENT.NUM_GPU", "DATA_LOADER.NUM_WORKER" and "DATA_LOADER.BATCH_SIZE" in 'FlareSeg/coarse_base_seg/config.yaml' and 'FlareSeg/fine_efficient_seg/config.yaml' files. You also need to set the 'nproc_per_node' in 'FlareSeg/coarse_base_seg/run.sh' file.
#### Coarse segmentation:
- Edit the 'FlareSeg/coarse_base_seg/config.yaml' and 'FlareSeg/coarse_base_seg/run.sh'
- Train coarse segmentation with the following command:
```bash
cd FlareSeg/coarse_base_seg
sh run.sh
```

#### Fine segmentation:
- Put the trained coarse model in the 'FlareSeg/model_weights/base_coarse_model/' folder.
- Edit the 'FlareSeg/fine_efficient_seg/config.yaml'.
- Edit the 'FlareSeg/fine_efficient_seg/run.py', set the 'tune_params' for different experiments.
- Train fine segmentation with the following command:
```bash
cd  FlareSeg/fine_efficient_seg
sh run.sh
```

### Inference:
- Put the trained models in the 'FlareSeg/model_weights/' folder.
- Run the inference with the following command:
```bash
sh predict.sh
```

## References
[1]  Z. Zhu, Y. Xia, W. Shen, E. Fishman, A. Yuille, A 3D coarse-to-fine framework for volumetric medical image segmentation,  2018 International conference on 3D vision (3DV), IEEE, 2018, pp. 682-690.

[2] F. Zhang, Y. Wang, H. Yang, Efficient Context-Aware Network for Abdominal Multi-organ Segmentation, arXiv preprint arXiv:2109.10601, (2021).

