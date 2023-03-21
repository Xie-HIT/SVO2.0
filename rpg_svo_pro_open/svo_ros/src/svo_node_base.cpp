#include "svo_ros/svo_node_base.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>
#include <svo/common/logging.h>
#include <vikit/params_helper.h>

namespace svo_ros {

void SvoNodeBase::initThirdParty(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();

  ros::init(argc, argv, "svo");
}

SvoNodeBase::SvoNodeBase()
: node_handle_(), private_node_handle_("~"),
  type_(SvoNodeBase::chooseType(private_node_handle_, use_multi_cam_)), // 选择模式：单目、双目、多目
        svo_interface_(type_, node_handle_, private_node_handle_, use_multi_cam_)
{
  if (svo_interface_.imu_handler_)
  {
    svo_interface_.subscribeImu();
  }
  svo_interface_.subscribeImage(); // SVO前端的线程
  svo_interface_.subscribeRemoteKey();
}

svo::PipelineType SvoNodeBase::chooseType(const ros::NodeHandle& pnh, bool& use_multi_cam)
{
    use_multi_cam = vk::param<bool>(pnh, "multi_cam", false);

    if(!use_multi_cam)
    {
        return vk::param<bool>(pnh, "pipeline_is_stereo", false) ?
               svo::PipelineType::kStereo : svo::PipelineType::kMono;
    } else {
        return svo::PipelineType::kMono_multi;
        // return svo::PipelineType::kArray;
    }

}

void SvoNodeBase::run()
{
  ros::spin();
  SVO_INFO_STREAM("SVO quit");
  svo_interface_.quit_ = true;
  SVO_INFO_STREAM("SVO terminated.\n");
}

}  // namespace svo_ros
