/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *  Copyright (c) 2016, ETH Zurich, Wyss Zurich, Zurich Eye
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file ceres/ReprojectionError.hpp
 * @brief Header file for the ReprojectionError class.
 * @author Stefan Leutenegger
 */

#pragma once

#include <memory>

#include <ceres/ceres.h>

#include "svo/common/camera.h"
#include "svo/ceres_backend/error_interface.hpp"
#include "svo/ceres_backend/pose_local_parameterization.hpp"
#include "svo/ceres_backend/reprojection_error_base.hpp"

namespace svo {
namespace ceres_backend {

/// \brief The 2D keypoint reprojection error.
class ReprojectionError : public ReprojectionErrorBase
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base class type.
  typedef ceres::SizedCostFunction<2, 7, 4, 7> base_t;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 2;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Default constructor.
  ReprojectionError(){}

  /// \brief Construct with measurement and information matrix
  /// @param[in] cameraGeometry The underlying camera geometry.
  /// @param[in] measurement The measurement.
  /// @param[in] information The information (weight) matrix.
  ReprojectionError(CameraConstPtr cameraGeometry,
                    const measurement_t& measurement,
                    const covariance_t& information);

  /// \brief Trivial destructor.
  virtual ~ReprojectionError()
  {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  /// \brief Set the underlying camera model.
  /// @param[in] cameraGeometry The camera geometry.
  void setCameraGeometry(
      CameraConstPtr camera_geometry)
  {
    CHECK(camera_geometry != nullptr);
    camera_geometry_ = camera_geometry;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  virtual void setInformation(const covariance_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  virtual const covariance_t& information() const
  {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  virtual const covariance_t& covariance() const
  {
    return covariance_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobians_minimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobians_minimal) const;

  inline void setDisabled(const bool disabled)
  {
    disabled_ = disabled;
  }

  inline void setPointConstant(const bool point_constant)
  {
    point_constant_ = point_constant;
  }

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  size_t parameterBlockDim(size_t parameter_block_idx) const
  {
    return base_t::parameter_block_sizes().at(parameter_block_idx);
  }

  /// @brief Residual block type as string
  virtual ErrorType typeInfo() const
  {
    return ErrorType::kReprojectionError;
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (2D) measurement.

  /// \brief The camera model:
  CameraConstPtr camera_geometry_;

  // weighting related
  covariance_t information_; ///< The 2x2 information matrix.
  covariance_t square_root_information_; ///< The 2x2 square root information matrix.
  covariance_t covariance_; ///< The 2x2 covariance matrix.

  bool disabled_ = false;
  bool point_constant_ = false;
};

}  // namespace ceres_backend
}  // namespace svo

#include <svo/ceres_backend/reprojection_error_impl.hpp>
