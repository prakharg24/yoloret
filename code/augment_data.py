import random
import tensorflow as tf
from PIL import Image, ImageDraw
import PIL
import torchvision.transforms.functional as F
import torch
import numpy as np

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def draw_PIL_image(image, boxes, labels):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy= boxes[i], outline="#000000")

    return new_image

def intersect(boxes1, boxes2):
    '''
        Find intersection of every box combination between two sets of box
        boxes1: bounding boxes 1, a tensor of dimensions (n1, 4)
        boxes2: bounding boxes 2, a tensor of dimensions (n2, 4)

        Out: Intersection each of boxes1 with respect to each of boxes2,
             a tensor of dimensions (n1, n2)
    '''
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy =  torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                        boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))

    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy , min=0)  # (n1, n2, 2)
    return inter[:, :, 0] * inter[:, :, 1]  #(n1, n2)

def parse_annot(annotation_line):
    annotations = annotation_line.split(" ")
    img_add = annotations[0]

    boxes_list = annotations[1:]
    num_boxes = len(boxes_list)//5

    boxes = list()
    labels = list()
    for i in range(num_boxes):
        shift = 5*i
        xmin = int(boxes_list[shift+0])
        ymin = int(boxes_list[shift+1])
        xmax = int(boxes_list[shift+2])
        ymax = int(boxes_list[shift+3])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(boxes_list[shift+4]))

    return {"image": img_add, "boxes": boxes, "labels": labels}

def cutout(image, boxes, labels, fill_val= 0, bbox_remove_thres= 0.4):
    '''
        Cutout augmentation
        image: A PIL image
        boxes: bounding boxes, a tensor of dimensions (#objects, 4)
        labels: labels of object, a tensor of dimensions (#objects)
        fill_val: Value filled in cut out
        bbox_remove_thres: Theshold to remove bbox cut by cutout

        Out: new image, new_boxes, new_labels
    '''
    if type(image) == PIL.Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    original_channel = image.size(0)

    new_image = image
    new_boxes = boxes
    new_labels = labels

    for _ in range(50):
        #Random cutout size: [0.15, 0.5] of original dimension
        cutout_size_h = random.uniform(0.15*original_h, 0.5*original_h)
        cutout_size_w = random.uniform(0.15*original_w, 0.5*original_w)

        #Random position for cutout
        left = random.uniform(0, original_w - cutout_size_w)
        right = left + cutout_size_w
        top = random.uniform(0, original_h - cutout_size_h)
        bottom = top + cutout_size_h
        cutout = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

        #Calculate intersect between cutout and bounding boxes
        overlap_size = intersect(cutout.unsqueeze(0), boxes)
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ratio = overlap_size / area_boxes
        #If all boxes have Iou greater than bbox_remove_thres, try again
        if ratio.min().item() > bbox_remove_thres:
            continue

        cutout_arr = torch.full((original_channel,int(bottom) - int(top),int(right) - int(left)), fill_val)
        new_image[:, int(top):int(bottom), int(left):int(right)] = cutout_arr

        #Create new boxes and labels
        boolean = ratio < bbox_remove_thres

        new_boxes = boxes[boolean[0], :]

        new_labels = labels[boolean[0]]

        return F.to_pil_image(new_image), new_boxes, new_labels

def mixup(image_info_1, image_info_2):
    '''
        Mixup 2 image

        image_info_1, image_info_2: Info dict 2 image with keys = {"image", "label", "box", "difficult"}
        lambd: Mixup ratio

        Out: mix_image (Temsor), mix_boxes, mix_labels, mix_difficulties
    '''
    lambd = random.uniform(0, 1)
    img1 = image_info_1["image"]    #Tensor
    img2 = image_info_2["image"]    #Tensor
    mixup_width = max(img1.shape[2], img2.shape[2])
    mix_up_height = max(img1.shape[1], img2.shape[1])

    mix_img = torch.zeros(3, mix_up_height, mixup_width)
    mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * lambd
    mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - lambd)

    mix_labels = torch.cat((image_info_1["label"], image_info_2["label"]), dim= 0)

    mix_boxes = torch.cat((image_info_1["box"], image_info_2["box"]), dim= 0)

    return F.to_pil_image(mix_img), mix_boxes, mix_labels


def load_mosaic(img_arr, box_arr, lbl_arr):

    boxes4 = []
    labels4 = []
    s = 640
    img4 = torch.zeros(3, s, s)
    xc, yc = [int(random.uniform(s * 0.25, s * 0.75)) for _ in range(2)]  # mosaic center x, y

    for i in range(len(img_arr)):
        h, w = img_arr[i].shape[1], img_arr[i].shape[2]

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s), min(s, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[:, y1a:y2a, x1a:x2a] = img_arr[i][:, y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        box_values = box_arr[i]
        if(len(box_values)>0):
            box_values[:, 0] = box_values[:, 0] + padw
            box_values[:, 1] = box_values[:, 1] + padh
            box_values[:, 2] = box_values[:, 2] + padw
            box_values[:, 3] = box_values[:, 3] + padh
        boxes4.append(box_values)
        labels4.append(lbl_arr[i])

    # Concat/clip labels
    if len(boxes4):
        boxes4 = np.concatenate(boxes4, 0)
        labels4 = np.concatenate(labels4, 0)
        np.clip(boxes4[:, :], 0, s, out=boxes4[:, :])  # use with random_affine

    return F.to_pil_image(img4), boxes4, labels4

do_cutout = True
do_mixup = True
do_mosaic = True
train_file = 'voc_train_14910.txt'
train_list = open(train_file).read().split("\n")[:-1]

list_len = len(train_list)

if(do_cutout):

    print("Doing Cutout Augmentation")
    for i in range(list_len//4):
        if(i%100==0):
            print(i)
        img1_ind = random.randint(0, list_len)
        img1_line = train_list[img1_ind]

        img1_info = parse_annot(img1_line)

        img1 = Image.open(img1_info["image"], mode= "r")
        img1 = img1.convert("RGB")
        boxes = torch.FloatTensor(img1_info["boxes"])
        labels = torch.LongTensor(img1_info["labels"])

        new_img, new_boxes, new_labels = cutout(img1, boxes, labels)
        new_boxes = new_boxes.numpy()
        new_labels = new_labels.numpy()

        img_fldr = "/".join(img1_info["image"].split("/")[:-1])

        new_file_name = img_fldr + "/" + "cutout_augment_" + str(i) + ".jpg"
        new_img.save(new_file_name)

        new_image_line = new_file_name
        for eb, el in zip(new_boxes, new_labels):
            for coord in eb:
                new_image_line += " " + str(int(coord))
            new_image_line += " " + str(int(el))

        train_list.append(new_image_line)

if(do_mixup):

    print("Doing Mixup Augmentation")
    for i in range(list_len//4):
        if(i%100==0):
            print(i)
        img1_ind = random.randint(0, list_len)
        img2_ind = random.randint(0, list_len)
        img1_line = train_list[img1_ind]
        img2_line = train_list[img2_ind]

        img1_info = parse_annot(img1_line)
        img2_info = parse_annot(img2_line)

        img1 = Image.open(img1_info["image"], mode= "r")
        img1 = img1.convert("RGB")
        boxes1 = torch.FloatTensor(img1_info["boxes"])
        labels1 = torch.LongTensor(img1_info["labels"])

        img1_dict = {"image": F.to_tensor(img1), "label": labels1, "box": boxes1}

        img2 = Image.open(img2_info["image"], mode= "r")
        img2 = img2.convert("RGB")
        boxes2 = torch.FloatTensor(img2_info["boxes"])
        labels2 = torch.LongTensor(img2_info["labels"])

        img2_dict = {"image": F.to_tensor(img2), "label": labels2, "box": boxes2}

        new_img, new_boxes, new_labels = mixup(img1_dict, img2_dict)
        new_boxes = new_boxes.numpy()
        new_labels = new_labels.numpy()

        img_fldr = "/".join(img1_info["image"].split("/")[:-1])

        new_file_name = img_fldr + "/" + "mixup_augment_" + str(i) + ".jpg"
        new_img.save(new_file_name)

        new_image_line = new_file_name
        for eb, el in zip(new_boxes, new_labels):
            for coord in eb:
                new_image_line += " " + str(int(coord))
            new_image_line += " " + str(int(el))

        train_list.append(new_image_line)

if(do_mosaic):

    print("Doing Mosaic Augmentation")
    for i in range(list_len//2):
        if(i%100==0):
            print(i)
        img_info_arr = []
        img_arr = []
        boxes_arr = []
        labels_arr = []
        for j in range(4):
            img_ind = random.randint(0, list_len)
            img_line = train_list[img_ind]

            img_info = parse_annot(img_line)
            img_info_arr.append(img_info)

            img = Image.open(img_info["image"], mode= "r")
            img = img.convert("RGB")
            boxes = torch.FloatTensor(img_info["boxes"])
            labels = torch.LongTensor(img_info["labels"])

            img_arr.append(F.to_tensor(img))
            boxes_arr.append(boxes)
            labels_arr.append(labels)

        new_img, new_boxes, new_labels = load_mosaic(img_arr, boxes_arr, labels_arr)

        img_fldr = "/".join(img1_info["image"].split("/")[:-1])

        new_file_name = img_fldr + "/" + "mosaic_augment_" + str(i) + ".jpg"
        new_img.save(new_file_name)

        new_image_line = new_file_name
        for eb, el in zip(new_boxes, new_labels):
            for coord in eb:
                new_image_line += " " + str(int(coord))
            new_image_line += " " + str(int(el))

        train_list.append(new_image_line)


final_str = "\n".join(train_list)

aug_train_file = 'voc_train_augment_' + str(len(train_list)) + ".txt"
aug_train_writer = open(aug_train_file, 'w')
aug_train_writer.write(final_str)
aug_train_writer.close()
