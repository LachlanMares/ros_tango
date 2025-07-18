# tango_ros

A ROS1 package for Tango as used here [https://podgorki.github.io/TANGO/data/TANGO_ICRA25.pdf], repository also includes robohop_ros.

## Repository Structure

```bash
ros_tango/
├── src/
│   ├── tango_ros/         # Tango package
│   └── robohop_ros/       # Robohop package
├── README.md
└── ...
```
## Installing ROS1 (Noetic)

These instructions assume Ubuntu 20.04 (Focal). Follow the official steps to install ROS Noetic:

### 1. Setup sources
```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

### 2. Setup keys
```bash
sudo apt update
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

### 3. Install ROS
```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```

### 4. Environment setup
```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 5. Install dependencies
```bash
sudo apt install python3-rosdep python3-rosinstall python3-vcstools python3-rosinstall-generator build-essential
```

### 6. Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

## Clone Repository
Clone the repository into catkin_ws:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:LachlanMares/ros_tango.git
cd ..
```

## Build
Source ROS and catkin_make package:
```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws/src/ros_tango
catkin_make
source devel/setup.bash
```

## Additional Model Files
### Fast-SAM 
These need to be copied into /ros_tango/src/robohop_ros/models

https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt
https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt

### Depth-Anything
These need to be copied into /ros_tango/src/robohop_ros/src/depth_anything/metric_depth/checkpoints

https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt
https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt

This needs to be copied into /ros_tango/src/robohop_ros/src/depth_anything/metric_depth/zoedepth/models/base_models/checkpoints

https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth

## Using robohop_ros

```bash
robohop_ros/
├── config/             # Contains parameters for ROS Nodes
├── images/
├── include/
├── launch/             # Launch files
├── maps/               # Stored graph files
│   └── LabBasicRun/ 
├── models/             # Fast-SAM models
├── msg/                # ROS custom .msg files
├── rviz/
├── scripts/            # Python3 ROS node scripts
├── src/
│   ├── depth_anything/ # Depth Anything Repository        
│   ├── LightGlue/       
│   └── saved_plans/    # Store previously processed maps as .pkl files    
├── srv/                # ROS Services 
├── README.md
└── ...
```

edit settings in robohop_parameters as minimum the image topic, robohop uses compressed images:
```bash
image_message_subscriber_topic: '/camera/front/decompressed_image'
compressed_image_message_subscriber_topic: '/camera/front/compressed_image'
```

source ros + launch robohop_ros with:
```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/src/tango_ros/devel.bash

roslaunch robohop_ros robohop.launch
```

if you have selected to load new maps in via service 
```bash
use_robohop_load_map_service: True
```
you will need to source ros and launch, to start
```bash
roslaunch robohop_ros robohop_load_map_via_service.launch
```

