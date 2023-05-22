- [Installation](#installation)
- [Preparing Dataset](#preparing-dataset)
- [Training a Model](#training-a-model)
- [Evaluating a Model](#evaluating-a-model)

---------------
## Installation

Setup a tensorflow environment,

```
conda create --name yoloret tensorflow-gpu==2.1.0
conda activate yoloret
```

Install additional packages

```
conda install tensorflow_model_optimization neural_structured_learning matplotlib opencv-python
```

--------------------
## Preparing Dataset
### VOC Dataset
Download and extract the VOC 2007 and 2012 dataset as shown below
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
```

We have provided label lists in the folder `data_paths` which can be directly adapted to prepare data for training and evaluation.
```
python update_voc_path.py "/path/to/VOC/dataset/"
```

### COCO Dataset
COCO dataset needs to be downloaded directly from the official website : https://cocodataset.org/#download

Download the `2017 Train Images`, `2017 Val Images` and `2017 Train/Val annotations`.

We have provided label lists in the folder `data_paths` which can be directly adapted to prepare data for training and evaluation.
```
python update_coco_path.py "/path/to/COCO/dataset/"
```

-------------------
## Training a Model
**Step 1 : Train with frozen transfer learning weights**
```
python main.py --mode=TRAIN --train_dataset=<train list> --val_dataset=<val list> --freeze --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```
**Step 2 : Fine tune the complete model**
```
python main.py --mode=TRAIN --train_dataset=<train list> --val_dataset=<val list> --train_unfreeze=<last checkpoint> --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```

Example,
```
python main.py --mode=TRAIN --train_dataset=voc_train_14910.txt --val_dataset=voc_val_1641.txt --freeze --classes_path=model_data/voc_classes.txt --backbone=EFFICIENTNETB3
python main.py --mode=TRAIN --train_dataset=voc_train_14910.txt --val_dataset=voc_val_1641.txt --train_unfreeze=<last checkpoint> --classes_path=model_data/voc_classes.txt --backbone=EFFICIENTNETB3
```

----------------------
## Evaluating a Model:
A trained detection model can be evaluated using the following code,
```
python main.py --mode=MAP --model=<final checkpoint> --test_dataset=<test list> --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```

We have provided a few model checkpoints for testing [here](). They can be used directly as follows

**MobileNetV2x0.75 (320x320) on VOC Dataset**
```
python main.py --mode=MAP --model=checkpoints/mobilenetv2x75_320_voc.h5  --test_dataset=voc_test_4952.txt --backbone=MOBILENETV2x75 --classes_path=model_data/voc_classes.txt --input_size=320 --input_size=320
```

**MobileNetV2x1.4 (320x320) on VOC Dataset**
```
python main.py --mode=MAP --model=checkpoints/mobilenetv2x14_320_voc.h5  --test_dataset=voc_test_4952.txt --backbone=MOBILENETV2x14 --classes_path=model_data/voc_classes.txt --input_size=320 --input_size=320
```

**EfficientNet-B3 (416x416) on COCO Dataset**
```
python main.py --mode=MAP --model=checkpoints/efficientnetb3_416_coco.h5  --test_dataset=coco_test_4952.txt --backbone=EFFICIENTNETB3 --classes_path=model_data/coco_classes.txt --input_size=416 --input_size=416
```

**EfficientNet-B3 (224x224) on COCO Dataset**
```
python main.py --mode=MAP --model=checkpoints/efficientnetb3_224_coco.h5  --test_dataset=coco_test_4952.txt --backbone=EFFICIENTNETB3 --classes_path=model_data/coco_classes.txt --input_size=224 --input_size=224
```

------------------
## Acknowledgement

This repository is adapted from https://github.com/fsx950223/mobilenetv2-yolov3
