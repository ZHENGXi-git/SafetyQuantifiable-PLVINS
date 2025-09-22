# Safety-quantifiable Line Feature-based Monocular Visual Localization with 3D Prior Map


### 1. Prerequisites

1.1 Ubuntu 20.04 and ROS Noetic

1.2 Dependency: Eigen3, Opencv4, and Ceres Solver.

### 2. Build VINS-Mono on ROS
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/HKUST-Aerial-Robotics/VINS-Mono.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

### 3. Build SQ-PLVINS on ROS

Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/ZHENGXi-git/SafetyQuantifiable-PLVINS.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

### 4. Run on EuRoC dataset

Download [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

run in the ~/catkin_ws/
```
    roslaunch vins_estimator euroc.launch
    roslaunch map_fusion euroc_tracking.launch 
    roslaunch vins_estimator vins_rviz.launch
    rosbag play YOUR_PATH_TO_DATASET/V1_02_medium.bag 
```

### 5. Related paper

```
[1] Qin, Tong, Peiliang Li, and Shaojie Shen. "Vins-mono: A robust and versatile monocular visual-inertial state estimator." IEEE Transactions on Robotics 34.4 (2018): 1004-1020.

[2] Yu, Huai, et al. "Monocular camera localization in prior lidar maps with 2d-3d line correspondences." 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2020.
```

### 6. Acknowledgement
This open-source is organized based on [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono.git). 
We would like to thank the authors of VINS for their generous sharing!

### 7. Licence
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

### 8. Citation
Zheng X, Wen W, Hsu L T. Safety-quantifiable line feature-based monocular visual localization with 3d prior map[J]. IEEE Transactions on Intelligent Transportation Systems, 2025.
