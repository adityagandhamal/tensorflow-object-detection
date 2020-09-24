### This project uses Tensorflow's Object Detection api to detect objects in a video.

#### Here's a snippet:
![tensorflow_obj_detection](https://user-images.githubusercontent.com/61016383/94112562-415c3200-fe63-11ea-98e8-367a4b5c1637.gif)

## Here is what you can do to use this project:
  
  - 1] Download and install [Anaconda](https://www.anaconda.com/)
  
  - 2] [Clone](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository) or Download the [tensorflow-object-detection-api](https://github.com/tensorflow/models) from Github
  
  - 3] Clone or Download this repo
  
  - 4] Open Anaconda Command Prompt and install the following packages for Windows:
      
         pip install tensorflow

         pip install opencv-python

         pip install Cython

         pip install contextlib2

         pip install pillow

         pip install lxml

         pip install tf_slim

  - 5] Copy and Paste `protoc.exe` file from this repo to the path `models-master/research`
  
  - 6] Open the Anaconda Command Prompt in `models-master\research` and copy and run the command written in `protoc_command.txt`
  
  - 7] Copy and Paste the file `tf2od_nyc.ipynb` into `models-master\research` directory
  
#### For more information regarding installation and configuration of the api head on to <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md>  


## About the project:
  - #### The model used in this project is [``SSD_MOBILENET_V1``](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz) trained on the [``COCO 2017 DATASET``](http://cocodataset.org/)
  
  These are the other models provided by TensorFlow for using its Object Detetcion API
  
  Model name                                                                                                                                                                  | Speed (ms) | COCO mAP | Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------: | :----------: | :-----:
[CenterNet HourGlass104 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz)                    | 70         | 41.9           | Boxes
[CenterNet HourGlass104 Keypoints 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz)                    | 76         | 40.0/61.4           | Boxes/Keypoints
[CenterNet HourGlass104 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz)               | 197       | 44.5           | Boxes
[CenterNet HourGlass104 Keypoints 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz)               | 211       | 42.8/64.5          | Boxes/Keypoints
[CenterNet Resnet50 V1 FPN 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz)     | 27         | 31.2           | Boxes
[CenterNet Resnet50 V1 FPN Keypoints 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz)     | 30         | 29.3/50.7         | Boxes/Keypoints
[CenterNet Resnet101 V1 FPN 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz)     | 34         | 34.2           | Boxes
[CenterNet Resnet50 V2 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz)     | 27         | 29.5           | Boxes
[CenterNet Resnet50 V2 Keypoints 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz)     | 30         | 27.6/48.2           | Boxes/Keypoints
[EfficientDet D0 512x512](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz)                                  | 39         | 33.6           | Boxes
[EfficientDet D1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)                                  | 54         | 38.4           | Boxes
[EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz)                                  | 67         | 41.8           | Boxes
[EfficientDet D3 896x896](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz)                                  | 95         | 45.4           | Boxes
[EfficientDet D4 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz)                              | 133         | 48.5           | Boxes
[EfficientDet D5 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz)                             | 222         | 49.7           | Boxes
[EfficientDet D6 1280x1280](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz)                             | 268         | 50.5           | Boxes
[EfficientDet D7 1536x1536](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz)                             | 325         | 51.2           | Boxes
[SSD MobileNet v2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)                                |19         | 20.2           | Boxes
[SSD MobileNet V1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 48        | 29.1           | Boxes
[SSD MobileNet V2 FPNLite 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)                | 22         | 22.2           | Boxes
[SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)                | 39         | 28.2           | Boxes
[SSD ResNet50 V1 FPN 640x640 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)                          | 46         | 34.3           | Boxes
[SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                      | 87         | 38.3           | Boxes
[SSD ResNet101 V1 FPN 640x640 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 57         | 35.6           | Boxes
[SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                    | 104        | 39.5           | Boxes
[SSD ResNet152 V1 FPN 640x640 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz)                        | 80         | 35.4           | Boxes
[SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)                    | 111        | 39.6           | Boxes
[Faster R-CNN ResNet50 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz)                 | 53         | 29.3           | Boxes
[Faster R-CNN ResNet50 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz)             | 65         | 31.0           | Boxes
[Faster R-CNN ResNet50 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz)               | 65         | 31.6           | Boxes
[Faster R-CNN ResNet101 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz)               |    55      | 31.8           | Boxes
[Faster R-CNN ResNet101 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz)           | 72         | 37.1           | Boxes
[Faster R-CNN ResNet101 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz)             | 77         | 36.6           | Boxes
[Faster R-CNN ResNet152 V1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz)               | 64         | 32.4           | Boxes
[Faster R-CNN ResNet152 V1 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz)           | 85         | 37.6           | Boxes
[Faster R-CNN ResNet152 V1 800x1333](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz)             | 101         | 37.4           | Boxes
[Faster R-CNN Inception ResNet V2 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz)             | 206         | 37.7           | Boxes
[Faster R-CNN Inception ResNet V2 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)             | 236         | 38.7           | Boxes
[Mask R-CNN Inception ResNet V2 1024x1024](http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz) |    301      | 39.0/34.6           | Boxes/Masks
[ExtremeNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/extremenet.tar.gz)                                                                         | --         | --           | Boxes


  - #### This project uses OpenCV to process the video frames and to return the video on which the final detections are made by the above mentioned detection model.
  
  - #### In this repo, I have also provided a python script ``tf2od_nyc.py`` as an alternative for the Jupyter Notebook file. 
