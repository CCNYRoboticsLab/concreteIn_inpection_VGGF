## Concrete Inpection Dataset and Baseline @ [CCNY Robotics Lab](https://ccny-ros-pkg.github.io/)

Authors: [Liang Yang](https://ericlyang.github.io/),  [Bing LI](https://robotlee2002.github.io/), [Wei LI](http://ccvcl.org/~wei/), Zhaoming LIU, Guoyong YANG, [Jizhong XIAO](http://www-ee.ccny.cuny.edu/www/web/jxiao/jxiao.html)


CCNY Concrete Structure Spalling and Crack database (CSSC) that aims to assist the civil inspection of performing in an automatical approach. In the first generate of our work, we mainly focusing on dataset creation and prove the concepts of innovativity. We provide the first complete the detailed dataset for concrete spalling and crack defects witht the help from Civil Engineers, where we also show our sincere thanks to the under-graduate student at Hostos Community College for their effort on data labeling. For our experiments, we deliever an UAV to perform field data-collection and inspection, and also perform a 3D semantic metric resconstructiont. 


### If you find this could be helpful for your project, please cite the following related papers:

[IROS 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). Deep Concrete Inspection Using Unmanned Aerial Vehicle Towards CSSC Database. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), One Page Abstract, [PDF](https://ericlyang.github.io/img/IROS2017/IROS2017.pdf).


[ROBIO 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). A Robotic System Towards Concrete Structure Spalling And Crack Database. 2017 IEEE Int. Conf. on Robotics and Biomimetics (ROBIO 2017), [Project](https://ericlyang.github.io/project/deepinspection/).


### The under going project

If you are interested in this project, please check the [project link](https://ericlyang.github.io/project/deepinspection/) and our current 3D semantic pixel-level reconstruction [project](https://ericlyang.github.io/project/robot-inspection-net/). Also, you can shoot [Liang Yang](https://ericlyang.github.io/) an email any time for other authors.


## 1. Prerequisites

The inspection network is trained based on [theano](http://deeplearning.net/software/theano/) and using [Lasagne Api](https://github.com/Lasagne/Lasagne). We performed training and evaluation on version:

>-  Tested Theano Version: '0.8.2'
>-  Lasagne version: '0.1'

If you change to the latest version, it should work with minor modification.


### 1.1 To enable a customized fine tuning, we  modified the Lasagne

Please follow the instruction to modify you lasagne to be able to train in your our computer.


### 1.2 The Computer

Our computer is a desktop with a GTX 1080 8G GPU inside and 32G memory, and it cann achieve 150 frames per second. We also tested on Dell Xps 15 with GTX 960M 2G GPU it can also achieve a 50 frames per second rate.

### 1.3 Other libraries

> - [Opencv](https://github.com/opencv/opencv) any verson is good, recomend the latest.
> - sudo pip install skimage
> - sudo pip install pickle
> - sudo pip install sklearn


## 2. Data set


## 3. Training and Testing





