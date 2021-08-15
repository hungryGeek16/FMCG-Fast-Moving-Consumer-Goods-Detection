# FMCG-Fast-Moving-Consumer-Goods-Detection

* This repository implements three detection algorithms on the given dataset and compares their performance. Details about preprocessing and different model architecture explanation has been discussed over here.
* Model which gave the best mAP score on test set was chosen. Please refer the documentation below to quickly start with data preparation, training and inference.
* Final model used for inference is FCOS(**F**ully **C**onvolutional **O**nes **S**tage object detection)
* Current implementation is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet) framework which is based on detectron2.
* Tested CPU version successfully on **Ubuntu 16.04** and GPU version on **Ubuntu 18.04**.

## 1. Initial setup

* Please make sure that python version is 3.7(conda env is recommended).Clone this repository and install requirements as shown below:
```bash
git clone 
pip3 install -r requirements.txt
pip3 install 'git+https://github.com/facebookresearch/detectron2.git' ##
```
* We'll call this repository folder as the ROOT folder.

## 2. Dataset Preparation

* This repository supports coco format datasets. Download dataset and annotations from [here](https://drive.google.com/file/d/1wldHgFHWn5ucErTWCytP2wozZdJWzoIF/view?usp=sharing) and keep them in ROOT folder. 

* After downloading the data, extract files from it. It's structure is shown below:
```bash
|--ROOT
     |--ShelfImages
           |--train(283 images)
           |--test(71 images)
     |--annotations.csv
```

* Run data_prep.py as shown below:

```bash
 python3 data_prep.py --test_dir ShelfImages/test --train_dir ShelfImages/train --csv_file annotations.csv
```
 
 * This will give out two files, namely instances_train.json and instances_test.json which are in coco format.

## 3. Model Training and inference

* Step 2 can be skipped if necessary, since instances_train.json and instances_test.json are already provided.

* Install adelaidet as given below:

```bash
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python3 setup.py build develop
```
* Train Model using the command below:

```bash
python3 train.py --cpu #For cpu training, remove "--cpu" to switch to gpu
```
* A colab version of training with gpu can be found [here]().(Recommended)

## 4. Inference

* Download model file from [here](https://drive.google.com/file/d/1-1DDG3GTOjV-kSrAIhVn5e8GU8Ofz-KP/view?usp=sharing) and keep it in the ROOT folder.

* Inference on cpu:

```bash
python3 inference.py --im /path/to/image
```
