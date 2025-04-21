# DDR-opt

DDR-opt is a universal Trajectory Optimization Framework for Differential Drive Robot Class.

The paper is published in T-ASE!

Please visit our project website [DDR-opt](https://zju-fast-lab.github.io/DDR-opt/).
If you find this work useful or interesting, please kindly give us a star ⭐, thanks! 😀

<p align="center">
    <img src="others/picture/head_figure_trajectory2.png" alt="Trajectory" width="70%">
</p>

## Quick Start
Compiling tests passed on ubuntu **18.04, and 20.04** with ros installed.
You can just execute the following commands one by one.
### Dependence:
``` bash
sudo apt install ros-noetic-tf2-sensor-msgs # or melodic
```

OSQP and OSQP-Eigen make it easier to modify parameters and are used to solve control problems under velocity and angular velocity control.
You can download them from the following two links:
- Download [osqp-v0.6.3-src.tar.gz](https://github.com/osqp/osqp/releases/tag/v0.6.3) or click [here](https://github.com/osqp/osqp/releases/download/v0.6.3/osqp-v0.6.3-src.tar.gz), and then follow the [installation instructions](https://osqp.org/docs/get_started/sources.html)
- Download [OSQP-eigen v0.8.1](https://github.com/robotology/osqp-eigen/releases/tag/v0.8.1).
```bash
cd osqp-eigen
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ../
make
sudo make install
```

**NOTE:** We may have forgotten other dependencies 😟, sorry! If you could provide missing dependencies, we would greatly appreciate it. 

### Build
``` bash
mkdir -p DDRopt_ws/src
cd DDRopt_ws/src
git clone git@github.com:ZJU-FAST-Lab/DDR-opt.git
cd ..
catkin build
```

### Run
You can run any of the following: 
``` bash
roslaunch plan_manager planner_nmpc.launch # for robots controlled by wheel speeds
roslaunch plan_manager planner_sim_unknown.launch # for planning in unknown space
roslaunch plan_manager planner_sim.launch # for robots controlled by linear and angular velocity
```

You can use `2D Nav Goal` to set goal point.  


## Citing
The method used in this software are described in the following paper (available on [IEEE](https://ieeexplore.ieee.org/document/10924228) and [arxiv](https://arxiv.org/abs/2409.07924v2))
```
@ARTICLE{zhang2024universaltrajectoryoptimizationframework,
  author={Zhang, Mengke and Chen, Nanhe and Wang, Hu and Qiu, Jianxiong and Han, Zhichao and Ren, Qiuyu and Xu, Chao and Gao, Fei and Cao, Yanjun},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={Universal Trajectory Optimization Framework for Differential Drive Robot Class}, 
  year={2025},
  volume={22},
  number={},
  pages={13030-13045},
  keywords={Robots;Mobile robots;Kinematics;Trajectory optimization;Planning;Robot kinematics;Computational modeling;Dynamics;Wheels;Tracking;Motion planning;trajectory optimization;differential drive robot class;nonholonomic dynamics},
  doi={10.1109/TASE.2025.3550676}}
```
