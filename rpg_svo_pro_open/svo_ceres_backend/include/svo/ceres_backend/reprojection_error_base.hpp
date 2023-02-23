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
 *  Created on: Jan 4, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Zurich Eye
 *********************************************************************************/

/**
 * @file ReprojectionErrorBase.hpp
 * @brief Header file for the ReprojectionErrorBase class.
 * @author Stefan Leutenegger
 */
#pragma once

#include <ceres/ceres.h>

#include "svo/ceres_backend/error_interface.hpp"

namespace svo {
namespace ceres_backend {

/// \brief Reprojection error base class.
class ReprojectionErrorBase :
    public ceres::SizedCostFunction<
    2 /* number of residuals */,
    7 /* size of first parameter */,
    4 /* size of second parameter */,
    7 /* size of third parameter (camera extrinsics) */>,
    public ErrorInterface
{
 public:
  /// \brief Measurement type (2D).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Covariance / information matrix type (2x2).
  typedef Eigen::Matrix2d covariance_t;

  virtual ~ReprojectionErrorBase() = default;

  /// \brief Set the measurement.
  /// @param[in] measurement The measurement vector (2d).
  virtual void setMeasurement(const measurement_t& measurement) = 0;

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix (2x2).
  virtual void setInformation(const covariance_t& information) = 0;

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector (2d).
  virtual const measurement_t& measurement() const = 0;

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix (2x2).
  virtual const covariance_t& information() const = 0;

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  virtual const covariance_t& covariance() const = 0;

};

} // namespace ceres_backend

} // namespace svo

