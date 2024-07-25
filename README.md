# PyTorch Implementation Of WS-DAN and SAC

## Introduction
这是SAC的PyTorch实现版本
## Environment
- Ubuntu 18.04, GTX 2080Ti 22G * 2, cuda 8.0
- Anaconda with Python=3.6.5, PyTorch=0.4.1, torchvison=0.2.1, etc.
- Some **third-party dependencies** may be installed with **pip** or **conda** when needed.

## Install

1. Clone 仓库
```
git clone https://github.com/17314796423/WS_DAN_SAC.git
```
2. 准备数据集
- 下载以下数据集. 

Dataset | Object | Category | Training | Testing
---|--- |--- |--- |---
[CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) | Bird | 200 | 5994 | 5794
[Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | Car | 100 | 6667 | 3333 
[fgvc-aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | Aircraft | 196 | 8144 | 8041
[Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) | Dogs | 120 | 12000 | 8580

- 抽取数据并整理为以下的目录格式:
```
Fine-grained
├── CUB_200_2011
│   ├── attributes
│   ├── bounding_boxes.txt
│   ├── classes.txt
│   ├── image_class_labels.txt
│   ├── images
│   ├── images.txt
│   ├── parts
│   ├── README
├── Car
│   ├── cars_test
│   ├── cars_train
│   ├── devkit
│   └── tfrecords
├── fgvc-aircraft-2013b
│   ├── data
│   ├── evaluation.m
│   ├── example_evaluation.m
│   ├── README.html
│   ├── README.md
│   ├── vl_argparse.m
│   ├── vl_pr.m
│   ├── vl_roc.m
│   └── vl_tpfp.m
├── dogs
│   ├── file_list.mat
│   ├── Images
│   ├── test_list.mat
│   └── train_list.mat
```
- 准备好 ./data 文件夹: 生成文件list txt (**使用 ./utils/convert_data.py**) 然后创建软链接. 
```
python utils/convert_data.py  --dataset_name bird --root_path .../Fine-grained/CUB_200_2011
```

```
├── data
│   ├── Aircraft -> /your_root_path/Fine-grained/fgvc-aircraft-2013b/data
│   ├── aircraft_test.txt
│   ├── aircraft_train.txt
│   ├── Bird -> /your_root_path/Fine-grained/CUB_200_2011
│   ├── bird_test.txt
│   ├── bird_train.txt
│   ├── Car -> /your_root_path/Fine-grained/Car
│   ├── car_test.txt
│   ├── car_train.txt
│   ├── Dog -> /your_root_path/Fine-grained/dogs
│   ├── dog_test.txt
│   └── dog_train.txt

```



## Usage

- Train

``` 
train_topk.py train --model-name inception --batch-size 24 --dataset bird --image-size 512 --input-size 448 --checkpoint-path checkpoint/bird --optim sgd --scheduler none --lr 0.001 --momentum 0.9 --weight-decay 1e-5 --workers 0 --parts 32 --epochs 80 --use-gpu --multi-gpu --gpu-ids 0,1 --resume-bap checkpoint/bird/non.pth.tar --resume checkpoint/bird/checkpoint.pth.tar --feature-extract True --prefix /home/ljy/Projects/WS_DAN_SAC/
```
一个简单的以后台使用sh的方式 `sh train_topk.sh` 或者用cmd运行在后台 `nohup sh train_topk.sh 1>train.log 2>error.log &`
- Test

```
python train_topk.py test --model-name inception --batch-size 24 --dataset bird --image-size 512 --input-size 448 --checkpoint-path checkpoint/bird/model_best.pth.tar --workers 0 --parts 32 --use-gpu --multi-gpu --gpu-ids 0,1 --feature-extract True --prefix /home/ljy/Projects/WS_DAN_SAC/
```

