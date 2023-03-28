// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/img_align/sparse_img_align.h>
#include <svo/abstract_bundle_adjustment.h>
#include <svo/direct/depth_filter.h>
#include <svo/tracker/feature_tracking_types.h>
#include <svo/initialization.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/reprojector.h>
#include <vikit/performance_monitor.h>

namespace svo {

FrameHandlerMono::FrameHandlerMono(
    const BaseOptions& base_options,
    const DepthFilterOptions& depth_filter_options,
    const DetectorOptions& feature_detector_options,
    const InitializationOptions& init_options,
    const ReprojectorOptions& reprojector_options,
    const FeatureTrackerOptions& tracker_options,
    const CameraBundle::Ptr& cam,
    bool use_multi_cam)
  : FrameHandlerBase(
      base_options, reprojector_options, depth_filter_options,
      feature_detector_options, init_options, tracker_options, cam, use_multi_cam)
{
//  // TEST ***
//  for(const auto& x: depth_filter_options.overlap)
//    for(const auto& y: x)
//      std::cout << y << ", ";
//  std::cout << std::endl;
}

UpdateResult FrameHandlerMono::processFrameBundle()
{
  UpdateResult res = UpdateResult::kFailure;
  if(stage_ == Stage::kTracking)
  {
    res = processFrame();
  }
  else if(stage_ == Stage::kInitializing)
  {
#if 1
    res = processFirstFrame();
#else
    // TODO (xie chen): leave to future work
    res = processFirstFrame_v2();
#endif
  }
  else if(stage_ == Stage::kRelocalization)
  {
    res = relocalizeFrame(Transformation(), reloc_keyframe_); // 重定位帧 reloc_keyframe_ 是上一个关键帧
  }
  return res;
}

void FrameHandlerMono::addImage(
    const cv::Mat& img,
    const uint64_t timestamp)
{
  addImageBundle({img}, timestamp);
}

UpdateResult FrameHandlerMono::processFirstFrame()
{
  if(!initializer_->have_depth_prior_)
  {
    initializer_->setDepthPrior(options_.init_map_scale); // 先验尺度
  }
  if(have_rotation_prior_)
  {
    VLOG(2) << "Setting absolute orientation prior";
    initializer_->setAbsoluteOrientationPrior(
          newFrame()->T_cam_imu().getRotation() * R_imu_world_);
  }

  /// 初始化
  const auto res = initializer_->addFrameBundle(new_frames_);

  // 积累初始化窗口，什么也不做
  if(res == InitResult::kTracking)
    return UpdateResult::kDefault;

  // TODO (xie chen): Add OLD multi-cameras to keyframe, map
  // make old frame keyframe
  initializer_->frames_ref_->setKeyframe();
  if(use_multi_cam_)
  {
    for(size_t i=0; i<initializer_->frames_ref_->frames_.size(); ++i)
    {
      const FramePtr& frame = initializer_->frames_ref_->frames_.at(i);

      if(i == 0)
        init_kf_bundle_id_ = frame->bundleId();

      frame->setKeyframe();
      if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
      {
        map_->addKeyframe(initializer_->frames_ref_->at(i),
                          bundle_adjustment_type_==BundleAdjustmentType::kCeres);
      }
      else if(bundle_adjustment_type_==BundleAdjustmentType::kGtsam)
      {
        CHECK(bundle_adjustment_)
        << "bundle_adjustment_type_ is kGtsam but bundle_adjustment_ is NULL";
        bundle_adjustment_->bundleAdjustment(initializer_->frames_ref_);
      }
      else
      {
        map_->addKeyframe(initializer_->frames_ref_->at(i), false);
      }
    }
  }
  else
  {
    initializer_->frames_ref_->at(0)->setKeyframe();
    if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
    {
      map_->addKeyframe(initializer_->frames_ref_->at(0),
                        bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    }
    else if(bundle_adjustment_type_==BundleAdjustmentType::kGtsam)
    {
      CHECK(bundle_adjustment_)
      << "bundle_adjustment_type_ is kGtsam but bundle_adjustment_ is NULL";
      bundle_adjustment_->bundleAdjustment(initializer_->frames_ref_);
    }
    else
    {
      map_->addKeyframe(initializer_->frames_ref_->at(0), false);
    }
  }

  // TODO (xie chen): Add NEW multi-cameras to keyframe, map, depth filter
  // make new frame keyframe
  new_frames_->setKeyframe();
  if(0) // TODO (xie chen): 不确定是否应将多相机地图点加入深度滤波器，因为之后尺度会调整
  {
    for(size_t i=0; i<newFrames().size(); ++i)
    {
      const FramePtr &newFrame = newFrames()[i];
      const FramePtr &lastFrameUnsafe = lastFramesUnsafe()[i];

      newFrame->setKeyframe();

      if(!frame_utils::getSceneDepth(newFrame,depth_median_[i], depth_min_[i], depth_max_[i]))
      {
        depth_min_[i] = 0.2; depth_median_[i] = 3.0; depth_max_[i] = 100;
      }
      VLOG(40) << "Current Frame Depth: " << "min: " << depth_min_[i]
               << ", max: " << depth_max_[i] << ", median: " << depth_median_[i];
      depth_filter_->addKeyframe(newFrame, depth_median_[i], 0.5*depth_min_[i],
                                 depth_median_[i]*1.5, std::make_pair(i+1, newFrames().size()));
      VLOG(40) << "Updating seeds in second frame using last frame...";
      depth_filter_->updateSeeds({ newFrame }, lastFrameUnsafe); // 对应相机更新深度滤波器

      // add frame to map
      map_->addKeyframe(newFrame,bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    }
  }
  else
  {
    newFrame()->setKeyframe();

    if(!frame_utils::getSceneDepth(newFrame(),depth_median_[0], depth_min_[0], depth_max_[0]))
    {
      depth_min_[0] = 0.2; depth_median_[0] = 3.0; depth_max_[0] = 100;
    }
    VLOG(40) << "Current Frame Depth: " << "min: " << depth_min_
             << ", max: " << depth_max_ << ", median: " << depth_median_;
    depth_filter_->addKeyframe(newFrame(), depth_median_[0], 0.5*depth_min_[0],
                               depth_median_[0]*1.5, std::make_pair(1, 1)); // 将当前帧选为关键帧
    VLOG(40) << "Updating seeds in second frame using last frame...";
    depth_filter_->updateSeeds({ newFrame() }, lastFrameUnsafe()); // 用上一帧去更新当前关键帧

    // add frame to map
    map_->addKeyframe(newFrame(),bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  }

  stage_ = Stage::kTracking;
  tracking_quality_ = TrackingQuality::kGood;
  initializer_->reset();
  VLOG(1) << "Init: Selected second frame, triangulated initial map.";
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerMono::processFirstFrame_v2()
{
  if(!initializer_->have_depth_prior_)
  {
    initializer_->setDepthPrior(options_.init_map_scale); // 先验尺度
  }
  if(have_rotation_prior_)
  {
    VLOG(2) << "Setting absolute orientation prior";
    initializer_->setAbsoluteOrientationPrior(
            newFrame()->T_cam_imu().getRotation() * R_imu_world_);
  }

  /// 初始化
  const auto res = initializer_->addFrameBundle_v2(new_frames_);

  // 积累初始化窗口，什么也不做
  if(res == InitResult::kTracking)
    return UpdateResult::kDefault;

  // make old frame keyframe
  initializer_->frames_ref_->setKeyframe();
  initializer_->frames_ref_->at(0)->setKeyframe();
  if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
  {
    map_->addKeyframe(initializer_->frames_ref_->at(0),
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  }
  else if(bundle_adjustment_type_==BundleAdjustmentType::kGtsam)
  {
    CHECK(bundle_adjustment_)
    << "bundle_adjustment_type_ is kGtsam but bundle_adjustment_ is NULL";
    bundle_adjustment_->bundleAdjustment(initializer_->frames_ref_);
  }
  else
  {
    map_->addKeyframe(initializer_->frames_ref_->at(0), false);
  }
  // make new frame keyframe
  newFrame()->setKeyframe();
  if(!frame_utils::getSceneDepth(newFrame(),depth_median_[0], depth_min_[0], depth_max_[0]))
  {
    depth_min_[0] = 0.2; depth_median_[0] = 3.0; depth_max_[0] = 100;
  }
  VLOG(40) << "Current Frame Depth: " << "min: " << depth_min_
  << ", max: " << depth_max_ << ", median: " << depth_median_;
  depth_filter_->addKeyframe(
          newFrame(), depth_median_[0], 0.5*depth_min_[0],
          depth_median_[0]*1.5, std::make_pair(1, 1)); // 将当前帧选为关键帧
  VLOG(40) << "Updating seeds in second frame using last frame...";
  depth_filter_->updateSeeds({ newFrame() }, lastFrameUnsafe()); // 用上一帧去更新当前关键帧

  // add frame to map
  map_->addKeyframe(newFrame(),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  stage_ = Stage::kTracking;
  tracking_quality_ = TrackingQuality::kGood;
  initializer_->reset();
  VLOG(1) << "Init: Selected second frame, triangulated initial map.";
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerMono::processFrame()
{
  VLOG(40) << "Updating seeds in overlapping keyframes...";
  // this is useful when the pipeline is with the backend,
  // where we should have more accurate pose at this moment
  // TODO (xie chen): 在重定位阶段就不要更新 Seed 了
  if(stage_ != Stage::kRelocalization)
  {
    if(use_multi_cam_)
    {
      for(size_t i=0; i<lastFrames().size(); ++i)
      {
        depth_filter_->updateSeeds(overlap_kfs_.at(i), lastFrames()[i]);
      }
    }
    else
    {
      depth_filter_->updateSeeds(overlap_kfs_.at(0), lastFrame());
    }
  }

  // ---------------------------------------------------------------------------
  // tracking

  // STEP 1: Sparse Image Align
  VLOG(40) << "===== Sparse Image Alignment =====";
  sparseImageAlignment();

  // STEP 2: Map Reprojection & Feature Align
  VLOG(40) << "===== Project Map to Current Frame =====";
  size_t n_total_observations = projectMapInFrame();
  if(n_total_observations < options_.quality_min_fts)
  {
    LOG(WARNING) << "Not enough feature after reprojection: "
                 << n_total_observations;
    return UpdateResult::kFailure;
  }

  // STEP 3: Pose & Structure Optimization
  // redundant when using ceres backend
  /// TODO (xie chen) 注意这里原本是不等于!=，但我们变成等于==，以提升一点精度
  if(bundle_adjustment_type_ == BundleAdjustmentType::kCeres)
  {
    VLOG(40) << "===== Pose Optimization =====";
    n_total_observations = optimizePose();
    if(n_total_observations < options_.quality_min_fts)
    {
      LOG(WARNING) << "Not enough feature after pose optimization."
                   << n_total_observations;
      return UpdateResult::kFailure;
    }

    optimizeStructure(new_frames_, options_.structure_optimization_max_pts, 5);
  }

  // TODO (xie chen): 检测 z 轴的运动是否足够，不够则在后端加入地面约束
  double new_frontend_z = new_frames_->get_T_W_B().getPosition()[2];
  double diff_z = fabs(new_frontend_z - last_frontend_z_);
  if(last_frontend_z_ == std::numeric_limits<double>::max() || newFrame()->bundleId() - init_kf_bundle_id_ < 10)
  {
    last_frontend_z_ = new_frontend_z;
    bundle_adjustment_->set_diff_z(100.0);
  }
  else if(diff_z >= 0.2)
  {
    bundle_adjustment_->set_z_flag(false);
    bundle_adjustment_->set_diff_z(diff_z); // diff_z 参与加权
  }
  else
  {
    bundle_adjustment_->set_z_flag(true);
    bundle_adjustment_->set_diff_z(diff_z);
  }

  // 跟踪不足的话，下面函数会报：Lost XX Features
  // return if tracking bad
  setTrackingQuality(n_total_observations);
  if(tracking_quality_ == TrackingQuality::kInsufficient)
    return UpdateResult::kFailure;

  // ---------------------------------------------------------------------------
  // select keyframe
  VLOG(40) << "===== Keyframe Selection =====";

  if(use_multi_cam_)
  {
    for(size_t i=0; i<newFrames().size(); ++i)
    {
      const FramePtr& newFrame = newFrames()[i];
      if(!frame_utils::getSceneDepth(newFrame, depth_median_[i], depth_min_[i], depth_max_[i])) // 每次都会重新覆盖所有相机的深度先验
      {
        depth_min_[i] = 0.2; depth_median_[i] = 3.0; depth_max_[i] = 100;
      }
      VLOG(40) << "Current Frame Depth: " << "min: " << depth_min_[i]
               << ", max: " << depth_max_[i] << ", median: " << depth_median_[i];
    }
  }
  else
  {
    if(!frame_utils::getSceneDepth(newFrame(),depth_median_[0], depth_min_[0], depth_max_[0]))
    {
      depth_min_[0] = 0.2; depth_median_[0] = 3.0; depth_max_[0] = 100;
    }
    VLOG(40) << "Current Frame Depth: " << "min: " << depth_min_[0]
             << ", max: " << depth_max_[0] << ", median: " << depth_median_[0];
  }

  initializer_->setDepthPrior(depth_median_[0]);
  initializer_->have_depth_prior_ = true;

//  // FIXME (xie chen): 后端普通帧只选择前两个相机
//  if(newFrames().size() > 2)
//  {
//    for(size_t i=2; i<newFrames().size(); ++i)
//    {
//      const FramePtr& newFrame = newFrames()[i];
//      newFrame->is_backend_imu_frame_ = false;
//    }
//  }

  // TODO (xie chen): 多相机的关键帧选取策略
  if(use_multi_cam_)
  {
    keyframe_candidates_.clear();
    for(size_t i=0; i<newFrames().size(); ++i)
    {
      const FramePtr& newFrame = newFrames()[i];

      if(need_new_kf_(newFrame->T_f_w_, i)/* TODO */
         && tracking_quality_ != TrackingQuality::kBad && stage_ != Stage::kRelocalization)
      {
        keyframe_candidates_.emplace_back(newFrame);
      }
      if(tracking_quality_ == TrackingQuality::kGood)
      {
        VLOG(40) << "Updating seeds in overlapping keyframes...";
        CHECK(!overlap_kfs_.empty());
      }
    }

    if(!keyframe_candidates_.empty())
    {
      for(const FramePtr& newFrame: keyframe_candidates_)
      {
        upgradeSeedsToFeatures(newFrame);
      }

      // TODO (xie chen): 包含 Seed 的跟踪，从多帧中筛选信息量最大的一帧
      if(newFrame()->bundleId() - init_kf_bundle_id_ > 10)
        greedy_select();

      std::cout << "update " << keyframe_candidates_.size() << " new keyframes" << std::endl;
    }
    else
    {
      std::cout << "update " << keyframe_candidates_.size() << " new keyframes" << std::endl;
      return UpdateResult::kDefault;
    }
  }
  else
  {
    // 关键帧选取策略，不需要关键帧会返回
    if(!need_new_kf_(newFrame()->T_f_w_, 0)
       || tracking_quality_ == TrackingQuality::kBad
       || stage_ == Stage::kRelocalization)
    {
      if(tracking_quality_ == TrackingQuality::kGood)
      {
        VLOG(40) << "Updating seeds in overlapping keyframes...";
        CHECK(!overlap_kfs_.empty());
        // now the seed is updated at the beginning of next frame
//      depth_filter_->updateSeeds(overlap_kfs_.at(0), newFrame());
      }
      return UpdateResult::kDefault;
    }
  }

  // TODO (xie chen)：加入关键帧的多相机之间应该互相知晓彼此
  if(use_multi_cam_)
  {
    for(const FramePtr& newFrame: keyframe_candidates_)
    {
      for(const FramePtr& keyframe_candidate: keyframe_candidates_)
      {
        newFrame->group_.insert(std::make_pair(keyframe_candidate->id(), keyframe_candidate));
      }
    }
  }

  // 若需要关键帧
  if(use_multi_cam_)
  {
    for(const FramePtr& newFrame: keyframe_candidates_)
    {
      newFrame->setKeyframe();
    }
  }
  else
  {
    newFrame()->setKeyframe();
  }
  VLOG(40) << "New keyframe selected.";

  // ---------------------------------------------------------------------------
  // new keyframe selected
  if (depth_filter_->options_.extra_map_points)
  {
    depth_filter_->sec_feature_detector_->resetGrid();
    OccupandyGrid2D map_point_grid(depth_filter_->sec_feature_detector_->grid_);
    reprojector_utils::reprojectMapPoints(
          newFrame(), overlap_kfs_.at(0),
          reprojectors_.at(0)->options_, &map_point_grid);
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    feature_detection_utils::mergeGrids(
          map_point_grid, &depth_filter_->sec_feature_detector_->grid_);
  }
  if(use_multi_cam_)
  {
//    for(const FramePtr& newFrame: keyframe_candidates_)
//    {
//      upgradeSeedsToFeatures(newFrame);
//    }
  }
  else
  {
    upgradeSeedsToFeatures(newFrame());
  }
  // init new depth-filters, set feature-detection grid-cells occupied that
  // already have a feature
  //
  // TODO: we should also project all seeds first! to make sure that we don't
  //       initialize seeds in the same location!!! this can be done in the
  //       depth-filter
  //
  {
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    setDetectorOccupiedCells(0, depth_filter_->feature_detector_);

  } // release lock
  if(use_multi_cam_)
  {
    for(size_t i=0; i<keyframe_candidates_.size(); ++i)
    {
      const FramePtr &newFrame = keyframe_candidates_[i];
      depth_filter_->addKeyframe(
              newFrame, depth_median_[i], 0.5*depth_min_[i],
              depth_median_[i]*1.5, std::make_pair(i+1, keyframe_candidates_.size()));
    }
  }
  else
  {
    depth_filter_->addKeyframe(
            newFrame(), depth_median_[0], 0.5*depth_min_[0],
            depth_median_[0]*1.5, std::make_pair(1, 1));
  }

  if(options_.update_seeds_with_old_keyframes)
  {
    VLOG(40) << "Updating seeds in current frame using last frame...";
    depth_filter_->updateSeeds({ newFrame() }, lastFrameUnsafe());
    VLOG(40) << "Updating seeds in current frame using overlapping keyframes...";
    for(const FramePtr& old_keyframe : overlap_kfs_.at(0))
      depth_filter_->updateSeeds({ newFrame() }, old_keyframe);
  }

  VLOG(40) << "Updating seeds in overlapping keyframes...";
//  depth_filter_->updateSeeds(overlap_kfs_.at(0), newFrame());

  // add keyframe to map
  if(use_multi_cam_)
  {
    for(const FramePtr& newFrame: keyframe_candidates_)
    {
      map_->addKeyframe(newFrame,
                        bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    }
  }
  else
  {
    map_->addKeyframe(newFrame(),
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  }

  // if limited number of keyframes, remove the one furthest apart
  if(options_.max_n_kfs > 2)
  {
    while(map_->size() > options_.max_n_kfs)
    {
      if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
      {
        // deal differently with map for ceres backend
        map_->removeOldestKeyframe();
      }
      else
      {
        FramePtr furthest_frame = map_->getFurthestKeyframe(newFrame()->pos());
        map_->removeKeyframe(furthest_frame->id());
      }
    }
  }
  return UpdateResult::kKeyframe;
}

// TODO (xie chen): 关键帧选择策略
void FrameHandlerMono::greedy_select()
{
  if(keyframe_candidates_.size() == 1)
    return;

  double max_score = -1.0;
  size_t max_score_id = -1;
  for(size_t i=0; i<keyframe_candidates_.size(); ++i)
  {
    FramePtr keyframe_candidate = keyframe_candidates_[i];
    double score = compute_score(keyframe_candidate);
    if(score > max_score)
    {
      max_score = score;
      max_score_id = i;
    }
  }

  for(size_t i=0; i<keyframe_candidates_.size(); ++i)
  {
    if(i == max_score_id)
      continue;
    keyframe_candidates_[i]->is_backend_keyframe_ = false;
  }
}

double FrameHandlerMono::compute_score(const FramePtr& frame)
{
  // FIXME (xie chen): 计算信息量，我也不知道最正确的计算方式是什么，下面的（启发式/玄学）方式暂且有效
  double score = 0.0;

  std::vector<FramePtr> active_keyframes;
  bundle_adjustment_->getAllActiveKeyframes(&active_keyframes);
  double depth_median, depth_min, depth_max;
  double depth_median_avg = 0.0;
  int count = 0;
  for(const auto& active_keyframe: active_keyframes)
  {
    if(frame_utils::getSceneDepth(active_keyframe,depth_median, depth_min, depth_max))
    {
      depth_median_avg += depth_median;
      ++count;
    }
  }
  depth_median_avg /= count;

  double lambda = 0.0;
  double lambda_z = 1.0;
  for(size_t i=0; i<frame->num_features_; ++i)
  {
    if(frame->landmark_vec_[i])
    {
      // Step1: 共视分数，倾向于被后端 Map 观测多的点
      /*
      size_t cnt = 0;
      for(const auto& active_keyframe: active_keyframes)
      {
        Eigen::Vector2d px_top_left(0.0, 0.0);
        Eigen::Vector3d f_top_left;
        active_keyframe->cam_->backProject3(px_top_left, &f_top_left);
        f_top_left.normalize();
        const Eigen::Vector3d z(0.0, 0.0, 1.0);
        const double min_cos = f_top_left.dot(z);

        Eigen::Vector3d xyz_cam = active_keyframe->T_f_w_ * frame->landmark_vec_[i]->pos();
        const double cur_cos = xyz_cam.normalized().dot(z);
        if (cur_cos > min_cos) // 可能被看见
        {
          ++cnt;
          score += cur_cos;
        }
      }
      score += lambda * std::pow(1.1, cnt);
       */

      // Step2: 深度分数，倾向于深度较近的点
      Position xyz_world = frame->landmark_vec_[i]->pos();
      Vector3d xyz_cam = frame->T_cam_world() * xyz_world;
      if(xyz_world[2] > 0) // FIXME (xie chen): 我不知道为什么深度能小于零 ?
      {
        if(static_cast<bool>(depth_median_avg))
        {
          double delta = xyz_world[2] - depth_median_avg;
          double score_z = delta<5.0? 100.0: 5.0/delta;
          score += lambda_z * score_z;
        }
        else
        {
          double score_z = xyz_world[2]<1.0? 1.0: 1.0/xyz_world[2];
          score += lambda_z * score_z;
        }
      }
    }
  }

  if(frame->getNFrameIndex() == 0 || frame->getNFrameIndex() == 1)
    score += 50.0;

  return score;
}

UpdateResult FrameHandlerMono::relocalizeFrame(
    const Transformation& /*T_cur_ref*/,
    const FramePtr& ref_keyframe)
{
  ++relocalization_n_trials_;
  if(ref_keyframe == nullptr)
    return UpdateResult::kFailure;

  VLOG_EVERY_N(1, 20) << "Relocalizing frame";
  // FIXME（xie chen）：当重定位时，下面只加入了一帧 ref_keyframe，导致后续某个 assert 会不支持多相机，解决方案是注释掉那个 assert（sparse_img_align.cpp line41）
  // TODO (xie chen)：多相机重定位时加入 multi-camera rig
  FrameBundle::Ptr ref_frame;
  if(use_multi_cam_)
  {
    std::vector<FramePtr> frames{ref_keyframe};
    for(const auto& pair: ref_keyframe->group_)
    {
      if(pair.first == ref_keyframe->id())
        continue;
      FramePtr other_keyframe = pair.second.lock();
      assert(other_keyframe->bundleId() == ref_keyframe->bundleId());
      frames.emplace_back(other_keyframe);
    }
    ref_frame = std::make_shared<FrameBundle>(frames,ref_keyframe->bundleId());
  }
  else
  {
    ref_frame = std::make_shared<FrameBundle>(std::vector<FramePtr>{ref_keyframe},ref_keyframe->bundleId());
  }
  last_frames_ = ref_frame;
  UpdateResult res = processFrame();
  if(res == UpdateResult::kDefault)
  {
    // Reset to default mode.
    stage_ = Stage::kTracking;
    relocalization_n_trials_ = 0;
    VLOG(1) << "Relocalization successful.";
  }
  else
  {
    // reset to last well localized pose
    newFrame()->T_f_w_ = ref_keyframe->T_f_w_;
  }
  return res;
}

void FrameHandlerMono::resetAll()
{
  if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
  {
    // with the ceres backend we have to make sure to initialize the scale
    backend_scale_initialized_ = false;
  }
  else
  {
    backend_scale_initialized_ = true;
  }
  resetVisionFrontendCommon();
}

FramePtr FrameHandlerMono::lastFrame() const
{
  return (last_frames_ == nullptr) ? nullptr : last_frames_->at(0);
}

std::vector<FramePtr>& FrameHandlerMono::lastFrames() const
{
  return last_frames_->frames_;
}

const FramePtr& FrameHandlerMono::newFrame() const
{
    return new_frames_->frames_[0];
}

const std::vector<FramePtr>& FrameHandlerMono::newFrames() const
{
    return new_frames_->frames_;
}


const FramePtr& FrameHandlerMono::lastFrameUnsafe() const
{
  return last_frames_->frames_[0];
}

const std::vector<FramePtr>& FrameHandlerMono::lastFramesUnsafe() const
{
  return last_frames_->frames_;
}

bool FrameHandlerMono::haveLastFrame() const
{
  return (last_frames_ != nullptr);
}

} // namespace svo
