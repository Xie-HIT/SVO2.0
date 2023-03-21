#pragma once

#include "svo_ros/svo_interface.h"

namespace svo_ros {

class SvoNodeBase {
 public:
  // Initializes glog, gflags and ROS.
  static void initThirdParty(int argc, char **argv);
  static svo::PipelineType chooseType(const ros::NodeHandle& pnh, bool& use_multi_cam);

  SvoNodeBase(); // （!）构造函数
  void run();

 private:
  ros::NodeHandle node_handle_;
  ros::NodeHandle private_node_handle_;
  svo::PipelineType type_; // 单目模式 或 双目模式
  bool use_multi_cam_;

 public:
    // 读取ROS参数，makeMono（读取相机的yaml配置文件/FrameHandlerMono）生成SVO前端
    // 产生可视化需要ROS广播的话题
    // 读取IMU参数
    // 开始SVO后端线程，后端需要ROS广播的话题
    // 读取回环参数
    // 调整状态机的状态为：开始
    svo::SvoInterface svo_interface_;
};

}  // namespace svo_ros
