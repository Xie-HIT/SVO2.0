#include "svo/tracker/feature_tracker.h"

#include <svo/common/camera.h>
#include <svo/common/container_helpers.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/direct/feature_alignment.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>

#include "svo/tracker/feature_tracking_utils.h"

namespace svo {

FeatureTracker::FeatureTracker(
    const FeatureTrackerOptions& options,
    const DetectorOptions& detector_options,
    const CameraBundlePtr& cams)
  : options_(options)
  , bundle_size_(cams->getNumCameras())
  , active_tracks_(bundle_size_)
  , terminated_tracks_(bundle_size_)
{
  for(size_t i = 0; i < cams->getNumCameras(); ++i)
  {
    detectors_.push_back(
          feature_detection_utils::makeDetector(
            detector_options, cams->getCameraShared(i)));
  }
}

void FeatureTracker::trackAndDetect(const FrameBundlePtr& nframe_kp1)
{
  size_t n_tracked = trackFrameBundle(nframe_kp1);
  if(n_tracked < options_.min_tracks_to_detect_new_features)
  {
    VLOG(4) << "Tracker: Detect new features";
    if(options_.reset_before_detection)
    {
      VLOG(4) << "Tracker: reset.";
      resetActiveTracks();

      VLOG(4) << "*** Tracker has " << active_tracks_.at(0).size();

      for(const FramePtr& frame : nframe_kp1->frames_)
        frame->clearFeatureStorage();
    }
    initializeNewTracks(nframe_kp1);
  }
}

size_t FeatureTracker::trackFrameBundle(const FrameBundlePtr& nframe_kp1)
{
  // Cleanup from previous tracking.
  resetTerminatedTracks();

  // TODO(cfo): Implement prediction when relative rotation is known.
  // TODO(cfo): Data structure could be simplified. We need only the first frame in track.
  for(size_t frame_index = 0; frame_index < bundle_size_; ++frame_index)
  {
    // active_tracks_ 是 FeatureTracks 的数组，数组长度为多目的数量
    // FeatureTracks 是 FeatureTrack 的数组，数组长度为（相应相机）特征点的数量
    // FeatureTrack 含有一个 track_id_ 和 一个 FeatureRefList 类型的对象
    // 其中 FeatureRefList 是 FeatureRef 的数组，FeatureRef 包含 帧、相机 和 feature 的索引，记录了track（与三维重建中的track含义一样）
    FeatureTracks& tracks = active_tracks_.at(frame_index); // 已有的 track，一开始是 0

    const FramePtr& cur_frame = nframe_kp1->at(frame_index);

    std::vector<size_t> remove_indices;
    Keypoints new_keypoints(2, tracks.size());
    Scores new_scores(tracks.size());
    TrackIds new_track_ids(tracks.size());
    size_t new_keypoints_counter = 0; // 记录跟踪上的总特征点个数

    for(size_t track_index = 0; track_index < tracks.size(); ++track_index)
    {
      FeatureTrack& track = tracks.at(track_index);
      const FeatureRef& ref_observation =
          (options_.klt_template_is_first_observation) ? track.front() : track.back();

      const ImgPyr& ref_pyr = ref_observation.getFrame()->img_pyr_;
      const ImgPyr& cur_pyr = cur_frame->img_pyr_;

      // TODO(cfo): make work for feature coordinates with subpixel reference patch!
      // Currently not a problem because feature detector returns integer pos.
      Eigen::Vector2i ref_px_level_0 = ref_observation.getPx().cast<int>(); // 参考帧下的特征点位置
      Keypoint cur_px_level_0 = track.back().getPx(); // 应该被当成光流法的初始值
      // 逆LK光流法
      bool success = feature_alignment::alignPyr2D(
            ref_pyr, cur_pyr,
            options_.klt_max_level, options_.klt_min_level, options_.klt_patch_sizes,
            options_.klt_max_iter, options_.klt_min_update_squared,
            ref_px_level_0, cur_px_level_0);
      if(success)
      {
        new_keypoints.col(new_keypoints_counter) = cur_px_level_0; // 在当前帧，跟踪第 new_keypoints_counter 个老特征点的位置 cur_px_level_0
        new_scores(new_keypoints_counter) =
            ref_observation.getFrame()->score_vec_[ref_observation.getFeatureIndex()]; // score 设成参考帧的
        new_track_ids(new_keypoints_counter) = track.getTrackId();
        track.pushBack(nframe_kp1, frame_index, new_keypoints_counter);
        ++new_keypoints_counter;
      }
      else
      {
        remove_indices.push_back(track_index); // 没跟踪上的特征点的索引加入 remove_indices
        terminated_tracks_.at(frame_index).push_back(track); // 没跟踪上的 track 加入 terminated_tracks_
      }
    }

    // TODO（xie chen）： 删除没跟踪上的 track：采用了拷贝的方式产生删除后的 tracks，从效率上来说不是太好
    // Remove keypoints to delete.
//    svo::common::container_helpers::eraseIndicesFromVector(remove_indices, &tracks);
    auto new_tracks = svo::common::container_helpers::eraseIndicesFromVector_DEPRECATED(tracks, remove_indices);
    tracks = new_tracks;

    // Insert new keypoints in frame.
    cur_frame->resizeFeatureStorage(new_keypoints_counter);
    cur_frame->px_vec_ = new_keypoints.leftCols(new_keypoints_counter);
    cur_frame->score_vec_ = new_scores.head(new_keypoints_counter);
    cur_frame->track_id_vec_ = new_track_ids.head(new_keypoints_counter);
    cur_frame->num_features_ = new_keypoints_counter;

    // Compute and normalize all bearing vectors.
    frame_utils::computeNormalizedBearingVectors(
          cur_frame->px_vec_, *cur_frame->cam(), &cur_frame->f_vec_);

    VLOG(4) << "Tracker: Frame-" << frame_index << " - tracked = " << new_keypoints_counter;
  }

  return getTotalActiveTracks();
}

size_t FeatureTracker::trackFrameBundle_v2(const FrameBundlePtr& nframe_kp1)
{
  // Cleanup from previous tracking.
  resetTerminatedTracks_v2();

  // 这些量之后会被赋值并写入当前相机，数组的长度为多目数量
  std::vector<Keypoints> new_keypoints;
  std::vector<Scores> new_scores;
  std::vector<TrackIds> new_track_ids;
  std::vector<size_t> new_keypoints_counter;
  for(size_t frame_index = 0; frame_index < bundle_size_; ++frame_index)
  {
    new_keypoints.emplace_back(Keypoints(2, active_tracks_v2_.size()));
    new_scores.emplace_back(Scores(active_tracks_v2_.size()));
    new_track_ids.emplace_back(TrackIds(active_tracks_v2_.size()));
    new_keypoints_counter.emplace_back(0);
  }

  std::vector<size_t> remove_indices;
  for(size_t i = 0; i < active_tracks_v2_.size(); ++i)
  {
    FeatureTrack_v2& track = active_tracks_v2_.at(i);
    const FeatureRef_v2& ref_observation = track.front();

    bool success = false;
    std::vector<size_t> frames_index;
    std::vector<size_t> features_index;
    for(size_t frame_index = 0; frame_index < bundle_size_; ++frame_index)
    {
      const FramePtr& cur_frame = nframe_kp1->at(frame_index);
      const ImgPyr& cur_pyr = cur_frame->img_pyr_;

      // 该 3D 点在首帧可能被多个相机看到，有一个相机跟踪上就认为跟踪成功
      for(size_t j=0; j<ref_observation.size(); ++j)
      {
        size_t ref_frame_index = ref_observation.getFramesIndex(j);
        size_t ref_feature_index = ref_observation.getFeaturesIndex(j);

        const FramePtr& ref_frame = ref_observation.getFrameBundle()->at(ref_frame_index);
        const ImgPyr& ref_pyr = ref_frame->img_pyr_;

        // TODO (xie chen): 在没有 3D 信息的情况下，不同相机检测的角点如何关联在一起
        // FIXME (xie chen): j 是在 track.front() 中的索引，对于 track.back()，也许会索引越界；另外 j 作为光流初始值也不合理
        Eigen::Vector2i ref_px_level_0 = ref_observation.getPx(j).cast<int>(); // 参考帧下的特征点位置
        Keypoint cur_px_level_0 = track.back().getPx(j); // 光流法的初始值

        // 逆LK光流法
        success = feature_alignment::alignPyr2D(
                ref_pyr, cur_pyr,
                options_.klt_max_level, options_.klt_min_level, options_.klt_patch_sizes,
                options_.klt_max_iter, options_.klt_min_update_squared,
                ref_px_level_0, cur_px_level_0);

        if(success)
        {
          new_keypoints.at(frame_index).col(new_keypoints_counter.at(frame_index)) = cur_px_level_0;
          new_scores.at(frame_index)(new_keypoints_counter.at(frame_index)) = ref_frame->score_vec_[ref_feature_index];
          new_track_ids.at(frame_index)(new_keypoints_counter.at(frame_index)) = track.getTrackId();
          frames_index.emplace_back(frame_index);
          features_index.emplace_back(new_keypoints_counter.at(frame_index));
          ++new_keypoints_counter.at(frame_index);
          break;
        }
      }
    }

    if(success) // 被 frame_bundle 看到就延续 track
    {
      track.pushBack(nframe_kp1, frames_index, features_index);
    }
    else // 未被 frame_bundle 看到就丢弃这个 track
    {
      remove_indices.push_back(i);
      terminated_tracks_v2_.push_back(track);
    }
  }

  // 删除没跟踪上的 track
  auto new_tracks = svo::common::container_helpers::eraseIndicesFromVector_DEPRECATED(active_tracks_v2_, remove_indices);
  active_tracks_v2_ = new_tracks;

  // 赋值给当前帧
  for(size_t frame_index = 0; frame_index < bundle_size_; ++frame_index)
  {
    const FramePtr& cur_frame = nframe_kp1->at(frame_index);

    // Insert new keypoints in frame.
    cur_frame->resizeFeatureStorage(new_keypoints_counter.at(frame_index));
    cur_frame->px_vec_ = new_keypoints.at(frame_index).leftCols(new_keypoints_counter.at(frame_index));
    cur_frame->score_vec_ = new_scores.at(frame_index).head(new_keypoints_counter.at(frame_index));
    cur_frame->track_id_vec_ = new_track_ids.at(frame_index).head(new_keypoints_counter.at(frame_index));
    cur_frame->num_features_ = new_keypoints_counter.at(frame_index);

    // Compute and normalize all bearing vectors.
    frame_utils::computeNormalizedBearingVectors(
            cur_frame->px_vec_, *cur_frame->cam(), &cur_frame->f_vec_);

    VLOG(4) << "Tracker: Frame-" << frame_index << " - tracked = " << new_keypoints_counter.at(frame_index);
  }

  return getTotalActiveTracks_v2();
}


size_t FeatureTracker::initializeNewTracks(const FrameBundlePtr& nframe)
{
  CHECK_EQ(nframe->size(), detectors_.size());
  CHECK_EQ(nframe->size(), active_tracks_.size());


  for(size_t frame_index = 0; frame_index < bundle_size_; ++frame_index)
  {
    // Detect features
    const FramePtr& frame = nframe->at(frame_index);
    detectors_.at(frame_index)->resetGrid();
    detectors_.at(frame_index)->grid_.fillWithKeypoints(frame->px_vec_);

    // Detect new features.
    Keypoints new_px;
    Levels new_levels;
    Scores new_scores;
    Gradients new_grads;
    FeatureTypes new_types;
    Bearings new_f;
    const size_t max_n_features = detectors_.at(frame_index)->grid_.size();
    detectors_.at(frame_index)->detect(
          frame->img_pyr_, frame->getMask(), max_n_features, new_px/* 新特征点的位置 */, new_scores,
          new_levels, new_grads, new_types);

    // Compute and normalize all bearing vectors.
    std::vector<bool> success;
    frame->cam()->backProject3(new_px, &new_f, &success); // 包含畸变矫正，不同的相机类型会有不同的子类实现
    for (const bool s : success) {
      CHECK(s);
    }

    new_f = new_f.array().rowwise() / new_f.colwise().norm().array();

    // Add features to frame.
    const size_t n_old = frame->num_features_;
    const size_t n_new = new_px.cols();
    frame->resizeFeatureStorage(n_old + n_new);
    frame->px_vec_.middleCols(n_old, n_new) = new_px;
    frame->f_vec_.middleCols(n_old, n_new) = new_f;
    frame->grad_vec_.middleCols(n_old, n_new) = new_grads;
    frame->score_vec_.segment(n_old, n_new) = new_scores;
    frame->level_vec_.segment(n_old, n_new) = new_levels;
    // TODO(cfo) frame->type_vec_
    frame->num_features_ = n_old+n_new;

    // Create a track for each feature
    FeatureTracks& tracks = active_tracks_.at(frame_index);
    tracks.reserve(frame->numFeatures());
    for(size_t feature_index = n_old; feature_index < frame->numFeatures(); ++feature_index)
    {
      const int new_track_id = PointIdProvider::getNewPointId();
      tracks.emplace_back(new_track_id);
      tracks.back().pushBack(nframe, frame_index, feature_index); // 新 track
      frame->track_id_vec_(feature_index) = new_track_id;
    }

    VLOG(4) << "Tracker: Frame-" << frame_index << " - detected = " << n_new
            << ", total = " << n_new + n_old;
  }

  return getTotalActiveTracks();
}

size_t FeatureTracker::initializeNewTracks_v2(const FrameBundlePtr& nframe)
{
  // TODO (xie chen): 在没有 3D 信息的情况下，不同相机检测的角点如何关联在一起
  // ...

  return getTotalActiveTracks();
}

const FeatureTracks& FeatureTracker::getActiveTracks(size_t frame_index) const
{
  CHECK_LT(frame_index, active_tracks_.size());
  return active_tracks_.at(frame_index);
}

size_t FeatureTracker::getTotalActiveTracks() const
{
  size_t i = 0;
  for(auto& tracks : active_tracks_)
    i += tracks.size();
  return i;
}

size_t FeatureTracker::getTotalActiveTracks_v2() const
{
  return active_tracks_v2_.size();
}


void FeatureTracker::getNumTrackedAndDisparityPerFrame(
    double pivot_ratio,
    std::vector<size_t>* num_tracked,
    std::vector<double>* disparity) const
{
  CHECK_NOTNULL(num_tracked);
  CHECK_NOTNULL(disparity);
  num_tracked->resize(bundle_size_);
  disparity->resize(bundle_size_);

  for(size_t i = 0; i < bundle_size_; ++i)
  {
    num_tracked->at(i) = active_tracks_[i].size();
    disparity->at(i) = feature_tracking_utils::getTracksDisparityPercentile(
          active_tracks_[i], pivot_ratio);
  }
}

void FeatureTracker::getNumTrackedAndDisparityPerFrame_v2(
        double pivot_ratio,
        size_t& num_tracked,
        double& disparity) const
{
  num_tracked = active_tracks_v2_.size();
  disparity = feature_tracking_utils::getTracksDisparityPercentile_v2(active_tracks_v2_, pivot_ratio);
}

FrameBundlePtr FeatureTracker::getOldestFrameInTrack(size_t frame_index) const
{
  CHECK_LT(frame_index, active_tracks_.size());
  const FeatureTrack& track = active_tracks_.at(frame_index).front();
  CHECK(!track.empty());
  return track.at(0).getFrameBundle();
}

void FeatureTracker::resetActiveTracks()
{
  for(auto& track : active_tracks_)
    track.clear();
}

void FeatureTracker::resetActiveTracks_v2()
{
  active_tracks_v2_.clear();
}

void FeatureTracker::resetTerminatedTracks()
{
  for(auto& track : terminated_tracks_) // 遍历每个相机的不要的 tracks
    track.clear();
}

void FeatureTracker::resetTerminatedTracks_v2()
{
  terminated_tracks_v2_.clear();
}

void FeatureTracker::reset()
{
  resetActiveTracks();
  resetTerminatedTracks();
  for(auto& detector : detectors_)
    detector->resetGrid();
}

} // namespace svo
