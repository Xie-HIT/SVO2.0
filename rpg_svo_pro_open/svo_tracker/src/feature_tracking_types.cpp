#include <svo/tracker/feature_tracking_types.h>
#include <svo/common/frame.h>
#include <numeric>

namespace svo {

// -----------------------------------------------------------------------------
FeatureRef::FeatureRef(
    const FrameBundlePtr& frame_bundle, size_t frame_index, size_t feature_index)
  : frame_bundle_(frame_bundle)
  , frame_index_(frame_index)
  , feature_index_(feature_index)
{
  CHECK_LT(frame_index_, frame_bundle_->size());
}

const Eigen::Block<Keypoints, 2, 1> FeatureRef::getPx() const
{
  return frame_bundle_->at(frame_index_)->px_vec_.block<2,1>(0, feature_index_);
}

const Eigen::Block<Bearings, 3, 1> FeatureRef::getBearing() const
{
  return frame_bundle_->at(frame_index_)->f_vec_.block<3,1>(0, feature_index_);
}

const FramePtr FeatureRef::getFrame() const
{
  return frame_bundle_->at(frame_index_);
}

// -----------------------------------------------------------------------------
FeatureRef_v2::FeatureRef_v2(
        const FrameBundlePtr& frame_bundle, const std::vector<size_t>& frames_index, const std::vector<size_t>& features_index)
        : frame_bundle_(frame_bundle)
        , frames_index_(frames_index)
        , features_index_(features_index)
{
  frame_bundle_id_ = frame_bundle->getBundleId();
}

/// id 是在 frames_index_ 或者 features_index_ 中的位置，具体哪一个相机是 frames_index_.at(id)
const Eigen::Block<Keypoints, 2, 1> FeatureRef_v2::getPx(size_t id) const
{
  return frame_bundle_->at(frames_index_.at(id))->px_vec_.block<2,1>(0, features_index_.at(id));
}


// -----------------------------------------------------------------------------
FeatureTrack::FeatureTrack(int track_id)
  : track_id_(track_id)
{
  feature_track_.reserve(10);
}

double FeatureTrack::getDisparity() const
{
  return (front().getPx() - back().getPx()).norm();
}

// -----------------------------------------------------------------------------
double FeatureTrack_v2::getDisparity() const
{
  const FeatureRef_v2& front_obs = front();
  const FeatureRef_v2& back_obs = back();
  size_t N1 = front_obs.size();
  size_t N2 = back_obs.size();
  assert(N1 > 0 && N2 > 0);

  std::vector<double> disparity; // TODO (xie chen)：仅考虑 track 在同一相机中的光流
  for(size_t i=0; i<N1; ++i)
  {
    size_t frame_index_front = front_obs.getFramesIndex(i);
    for(size_t j=0; j<N2; ++j)
    {
      size_t frame_index_back = back_obs.getFramesIndex(j);
      if(frame_index_front == frame_index_back)
      {
        disparity.emplace_back((front_obs.getPx(i) - back_obs.getPx(j)).norm());
        break;
      }
    }
  }

  return std::accumulate(disparity.begin(), disparity.end(), 0.0) / static_cast<int>(disparity.size());
}

} // namespace svo
