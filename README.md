<div align="center">

[![stars - badge-generator](https://img.shields.io/github/stars/prakharg24/yoloret)](https://github.com/prakharg24/yoloret)
[![forks - badge-generator](https://img.shields.io/github/forks/prakharg24/yoloret)](https://github.com/prakharg24/yoloret)
[![License](https://img.shields.io/badge/License-MIT-red)](https://github.com/prakharg24/yoloret/blob/main/LICENSE)
[![issues](https://img.shields.io/github/issues/prakharg24/yoloret)](https://github.com/prakharg24/yoloret/issues)
[![issues closed](https://img.shields.io/github/issues-closed/prakharg24/yoloret)](https://github.com/prakharg24/yoloret/issues)
[![last commit](https://img.shields.io/github/last-commit/prakharg24/yoloret)](https://github.com/prakharg24/yoloret)

</div>

### YOLO-ReT: Towards High Accuracy Real-time Object Detection on Edge GPUs

Prakhar Ganesh, Yao Chen, Yin Yang, Deming Chen, Marianne Winslett, **WACV 2022** \
[\[Paper\]](https://openaccess.thecvf.com/content/WACV2022/html/Ganesh_YOLO-ReT_Towards_High_Accuracy_Real-Time_Object_Detection_on_Edge_GPUs_WACV_2022_paper.html) [\[PDF\]](https://prakharg24.github.io/files/yolo_ret.pdf) [\[Slides\]](https://prakharg24.github.io/files/yolo_ret_slides.pdf) [\[Poster\]](https://prakharg24.github.io/files/yolo_ret_poster.pdf) [\[Video\]](https://drive.google.com/file/d/18j-OdX7ChcvLbNW0jO-qGbODRqZmDiX9/view)

![YOLO-ReT Architecture](https://github.com/prakharg24/yoloret/assets/20368770/75d6ee9f-b5d4-4d96-b37a-66551c466ec3)

> **Abstract**: Performance of object detection models has been growing rapidly. However, in order to map deep neural network (DNN) based object detection models to edge devices, one typically needs to compress such models significantly, thus compromising the model accuracy. In this project, we propose a novel edge GPU friendly module for multi-scale feature interaction by exploiting missing combinatorial connections between various feature scales in existing state-of-the-art methods. Additionally, we propose a novel transfer learning backbone adoption inspired by the changing translational information flow across various tasks, designed to complement our feature interaction module and together improve both accuracy as well as execution speed on various edge GPU devices available in the market. \
Moreover, in our evaluation, we compare the latency of our proposed model on actual devices, rather than relying only on model size or FLOPs as an indicator of performance. This allows us to better understand the real-world applicability of our model and make more informed decisions about its use. We compare the runtime FPS of various models on Jetson Nano, Jetson Xavier NX and Jetson AGX Xavier.


### Setup and Reproducibility

Please refer to the [README](code/README.md) inside the folder `code` for details on code setup and reproducibility.

### Citation

If you find our paper and/or code useful, please consider citing our work.

```
@inproceedings{ganesh2022yoloret,
  title={{YOLO-ReT}: Towards High Accuracy Real-time Object Detection on Edge {GPU}s},
  author={Ganesh, Prakhar and Chen, Yao and Yang, Yin and Chen, Deming and Winslett, Marianne},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2022},
  organization={IEEE}
}
```
