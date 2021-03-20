# CS231a Project Milestone: Monocular Depth Estimation for Mobile and Embedded Decvices

This repository contains code for my project for CS231a. Some network layers, loss functions as well as the dataloaders and preprocessing scripts are taken from the official **[FastDepth repository]( https://github.com/dwofk/fast-depth)**. Adjustments have been made based on the FastDepth code in [this implementation](https://github.com/tau-adl/FastDepth).

The design of the learner and the callback based system is from the **[FastAI library and course](https://github.com/fastai/course-v3)**, an open-source online course which I participated in. The learner from the course has been adapted to work with this dataset.

The custom MobileNetV2 + NNConv5 architecture is defined at the end of the `models.py` file. 

Bulk of the training experiments are under Experiments.ipynb. Please note that these notebooks make use of the NYU Depth V2 dataset. The dataset can be downloaded from [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat).

# Getting Started

## Setting up Data
* If using the NYU Depth V2 Labeled subset, please download it from [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and **configure the path to it in config.py** (`subset_datapath`).
* If making use of the full NYU Depth V2 Dataset as curated by the FastDepth paper, please download it from [here](http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz). Please note that this data set is 32G when compressed and 33G uncompressed. Uncompress this data as shown below and **then set the path to the data in config.py** (`datapath`).
```
mkdir data; cd data
wget http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz
tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
cd ..
```
* If training on the unaltered MobileNetSkipAdd architecture from FastDepth paper, please download the pretrained mobilenet model from [this link](http://datasets.lids.mit.edu/fastdepth/imagenet/). Place the .pth.tar file directly under the `imagenet/` folder.

## Training the model
* Install all conda dependencies from the environment.yml file.
* Configure training parameters in the config.py file.
* Run the following command to begin training.
```
python train.py
```

## Exporting the trained model
* Once the model has been trained, checkpoint .pth files will be saved in the folder specified in the learner.
* Use the ModelExport.ipynb file to interactively export to ONNX.

## Testing the models
* Checkpoints for full model training are available at [this link](https://drive.google.com/drive/folders/1sO0T16W6trusJk8UIDSeFhBjB2XfBWHa?usp=sharing). ```all_data_91.pth``` is the best performing model. The corresponding ONNX exported model is ```depth91.onnx```. A checkpoint can be loaded as follows:
```
from models import *
model = MobileNetV2SkipAdd(pretrained=True, interpolation='bilinear')
model.load_state_dict(torch.load('all_data_91.pth', map_location=torch.device('cpu'))['state_dict'])
```

* I've created a Snapchat Community Lens using [Snap Lens Studio](https://lensstudio.snapchat.com/), an editor for building community AR effects for Snapchat. This allows for easy testing of the model on a range of mobile devices. To test this lens, **please download the [Snapchat](https://www.snapchat.com/)** application on your mobile phone, create an account/login to your account and then scan the snapcode below by **pressing and holding the snapcode below on the camera-screen of the app**.

![snapcode](snapcode.png)

* Lens studio project which visualizes depth can be found [here](https://drive.google.com/drive/folders/1e-ozhOI_9UfAhMbdOQw27IJ8r2U82xlY?usp=sharing). In order to open it, please download [Snap Lens Studio](https://lensstudio.snapchat.com/download/) first. Then download the contents of the drive folder. Open the .lsproj file in Lens Studio. Note that I had to assume depth up to 20m for visualization.
