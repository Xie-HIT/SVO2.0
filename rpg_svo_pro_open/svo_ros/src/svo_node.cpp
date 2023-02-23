#include "svo_ros/svo_node_base.h"

int main(int argc, char **argv)
{
  // log设置，ros::init
  svo_ros::SvoNodeBase::initThirdParty(argc, argv);

  // ros::NodeHandle，运行模式（单目/双目）
  // 初始化 svo_interface_ 成员变量：
  //    - 读取ROS参数，makeMono（读取相机的yaml配置文件/FrameHandlerMono）生成SVO前端
  //    - 产生可视化需要ROS广播的话题
  //    - 读取IMU参数
  //    - 【开始SVO后端线程】，后端需要ROS广播的话题
  //    - 读取回环参数
  //    - 调整状态机的状态为：开始
  // 订阅IMU，图像和键盘控制指令：在图像回调函数里进行【SVO前端】
  svo_ros::SvoNodeBase node;

  // ros::spin()：只对主线程（键盘控制订阅）有效
  node.run();
}
