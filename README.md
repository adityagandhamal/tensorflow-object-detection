### This project uses Tensorflow's Object Detection api to detect objects in a video.

#### Here's a snippet:
![tensorflow_obj_detection](https://user-images.githubusercontent.com/61016383/94112562-415c3200-fe63-11ea-98e8-367a4b5c1637.gif)

## Here is what you can do to use this project:
  
  - Download and install [Anaconda](https://www.anaconda.com/)
  
  - [Clone](https://docs.github.com/en/enterprise/2.13/user/articles/cloning-a-repository) or Download the [tensorflow-object-detection-api](https://github.com/tensorflow/models) from Github
  
  - Clone or Download this repo
  
  -  Open Anaconda Command Prompt and install the these packages for Windows:
      `pip install tensorflow1`,
      `pip install opencv-python`,
      `pip install Cython`,
      `pip install contextlib2`,
      `pip install pillow`,
      `pip install lxml`,
      `pip install tf_slim`
   
  - Copy and Paste `protoc.exe` file from this repo to the path `models-master/research`
  
  - Open the Anaconda Command Prompt in `models-master\research` and copy and run the command written in `protoc_command.txt`
  
  - Copy and Paste the file `tf2od_nyc.ipynb` into `models-master\research` directory
  
#### For more information regarding installation and configuration of the api head on to <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md>  
