# OpticalFromDepth
This repository contains the source code for our paper:

Skin the sheep not only once:
Reusing Various Depth Datasets to Drive the Learning of Optical Flow</br>
Sheng-Chi Huang Wei-Chen Chiu<br/>

<img src="teaser.png">

## Introduction
        Optical flow estimation is crucial for various applications in vision and robotics. As the difficulty of collecting ground truth optical flow in real-world scenarios, most of the existing methods of learning optical flow still adopt synthetic dataset for supervised training or utilize photometric consistency across temporally adjacent video frames to drive the unsupervised learning, where the former typically has issues of generalizability while the latter usually performs worse than the supervised ones. To tackle such challenges, we propose to leverage the geometric connection between optical flow estimation and stereo matching (based on the similarity upon finding pixel correspondences across images) to unify various real-world depth estimation datasets for generating supervised training data upon optical flow. Specifically, we turn the monocular depth datasets into stereo ones via synthesizing virtual disparity, thus leading to the flows along the horizontal direction; moreover, we introduce virtual camera motion into stereo data to produce additional flows along the vertical direction. Furthermore, we propose applying geometric augmentations on one image of an optical flow pair, encouraging the optical flow estimator to learn from more challenging cases. Lastly, as the optical flow maps under different geometric augmentations actually exhibit distinct characteristics, an auxiliary classifier which trains to identify the type of augmentation from the appearance of the flow map is utilized to further enhance the learning of the optical flow estimator. Our proposed method is general and is not tied to any particular flow estimator, where extensive experiments based on various datasets and optical flow estimation models verify its efficacy and superiority.

## Installation

Create a virtual environment for this project.
```Shell
conda create --name OpticalFromDepth python=3.9
conda activate OpticalFromDepth
```

Clone this repo and install required packages, the code was developed with PyTorch 1.12.1 and Cuda 11.3.
```Shell
git clone https://github.com/AegeanKI/experiment
cd experiment
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirement.txt
```

Compile `my_cuda_loffsi` module, which is written in C, to handle warping operation.
```Shell
cd my_cuda_loffsi
python setup.py install
```

## Preprocessing
We use `DIML` as sample, you can also use `filted_ReDWeb`.

```Shell
python preprocess_continuous.py --dataset DIML \
                                --gpu 0 \
                                --split 1 \
                                --split_id 0
```

These parameters are:
- `dataset`: preprocess specific dataset
- `gpu`: preprocess dataset on specific gpu
- `split` and `split_id`: only preprocess part of dataset
- `subdir`: set the output subdirectory

## Datasets

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [Sintel](http://sintel.is.tue.mpg.de/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [DIML](https://dimlrgbd.github.io/#main)
- [ReDWeb](https://sites.google.com/site/redwebcvpr18/)

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── ReDWeb
        ├── Imgs
        ├── RDs
    ├── DIML
        ├── test
        ├── train
    ├── AugmentedDatasets
        ├── ReDWeb
        ├── DIML
```

## Training
We use the RAFT model as sample.

```Shell
cd adjusted_RAFT
python -u train.py --name adjusted_raft --stage mixed --validation kitti --gpus 0 \
                   --num_steps 120000 --batch_size 8 --lr 0.0025 --val_freq 10000 \
                   --mixed_precision \
                   --is_first_stage \
                   --add_classifier \
                   --classifier_checkpoint_timestamp 1677566045.275271 \
                   --classifier_checkpoint_train_acc 0.805 \
                   --classifier_checkpoint_test_acc 0.802 \
                   --classify_loss_weight_init 1 \
                   --classify_loss_weight_increase -0.00002 \
                   --max_classify_loss_weight 1 \
                   --min_classify_loss_weight 0
```

These parameters are:
- `add_classifier`: enable classifier while training optical flow estimator
- `classifier_checkpoint_timestamp`: choose classifier directory (for classifier setting and checkpoints)
- `classifier_checkpoint_train_acc` and `classifier_checkpoint_test_acc`: choose classifier checkpoint
- `classify_loss_weight_init` and `classify_loss_weight_increase`: adjust the impact of classifier (linearly)
- `max_classify_loss_weight` and `min_classify_loss_weight`: set the upper and the lower bound of the impact of the classifier













<!--
```
@article{,
  title={Deep monocular depth estimation leveraging a large-scale outdoor stereo dataset},
  author={Cho, Jaehoon and Min, Dongbo and Kim, Youngjung and Sohn, Kwanghoon},
  journal={Expert Systems with Applications},
  volume={178},
  year={2021},
}
```



## Demos
Pretrained models can be downloaded by running
```Shell
./download_models.sh
```
or downloaded from [google drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model=models/raft-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you will need to download the required datasets. 
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)


By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder


## Evaluation
You can evaluate a trained model using `evaluate.py`
```Shell
python evaluate.py --model=models/raft-things.pth --dataset=sintel --mixed_precision
```

## Training
We used the following training schedule in our paper (2 GPUs). Training logs will be written to the `runs` which can be visualized using tensorboard
```Shell
./train_standard.sh
```

If you have a RTX GPU, training can be accelerated using mixed precision. You can expect similiar results in this setting (1 GPU)
```Shell
./train_mixed.sh
```

## (Optional) Efficent Implementation
You can optionally use our alternate (efficent) implementation by compiling the provided cuda extension
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and running `demo.py` and `evaluate.py` with the `--alternate_corr` flag Note, this implementation is somewhat slower than all-pairs, but uses significantly less GPU memory during the forward pass.
-->

