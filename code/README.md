Complete source code for training our YOLO RED model.

# Requirements

Please check [requirements.txt](requirements.txt) for the major library requirements and

# Preparing Dataset
## Download VOC Dataset
Download and extract the VOC 2007 and 2012 dataset as shown below

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
```

## Download COCO Dataset
COCO dataset needs to be downloaded directly from the official website : https://cocodataset.org/#download

Download the `2017 Train Images`, `2017 Val Images` and `2017 Train/Val annotations`

## Format label list
We have provided label lists in the folder `data_paths` which can be directly adapted for use.

For example, if the VOC Dataset is downloaded and extracted at absolute location `/home/VOC/`, then the following script will automatically create the required label lists
```
python update_voc_path.py "/home/VOC/"
```

Similarly, the COCO label lists can also be created
```
python update_coco_path.py "/home/COCO/"
```

# Architecture Definition

The architecture definition of the RFCR module as well as various variations of the YOLO-RED family are provided in the file `yolo3/model.py`.

# Usage
## Get help info:
```
python main.py --help
```

## Train
### Step 1 : Train with frozen transfer learning weights
```
python main.py --mode=TRAIN --train_dataset=<train list> --val_dataset=<val list> --freeze --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```
### Step 2 : Fine tune complete model
```
python main.py --mode=TRAIN --train_dataset=<train list> --val_dataset=<val list> --train_unfreeze=<last checkpoint> --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```

## Test:
```
python main.py --mode=MAP --model=<final checkpoint> --test_dataset=<test list> --classes_path=<class names list> --backbone=<MOBILENETV2x75|MOBILENETV2x14|EFFICIENTNETB3>
```

Note : `<class names list>` are provided in the folder `model_data` as `voc_classes.txt` and `coco_classes.txt`, depending on the training dataset.

# Testing Existing Models

We have provided a few model checkpoints for testing. They can be used directly as follows

### MobileNetV2x0.75 (320x320) on VOC Dataset
```
python main.py --mode=MAP --model=checkpoints/mobilenetv2x75_320_voc.h5  --test_dataset=voc_test_4952.txt --backbone=MOBILENETV2x75 --classes_path=model_data/voc_classes.txt --input_size=320 --input_size=320
```

### MobileNetV2x1.4 (320x320) on VOC Dataset
```
python main.py --mode=MAP --model=checkpoints/mobilenetv2x14_320_voc.h5  --test_dataset=voc_test_4952.txt --backbone=MOBILENETV2x14 --classes_path=model_data/voc_classes.txt --input_size=320 --input_size=320
```

### EfficientNet-B3 (416x416) on COCO Dataset
```
python main.py --mode=MAP --model=checkpoints/efficientnetb3_416_coco.h5  --test_dataset=coco_test_4952.txt --backbone=EFFICIENTNETB3 --classes_path=model_data/coco_classes.txt --input_size=416 --input_size=416
```

### EfficientNet-B3 (224x224) on COCO Dataset
```
python main.py --mode=MAP --model=checkpoints/efficientnetb3_224_coco.h5  --test_dataset=coco_test_4952.txt --backbone=EFFICIENTNETB3 --classes_path=model_data/coco_classes.txt --input_size=224 --input_size=224
```
