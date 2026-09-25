#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"
#include "cora_symforce/sesync_matrices.hpp"

namespace cora_symforce {

struct SimplifiedProblem {
  Eigen::MatrixXd Pi;
  Eigen::MatrixXd Q;
};

SimplifiedProblem buildSimplifiedProblem(
    const std::vector<RelativePoseMeasurement>& measurements,
    std::size_t num_poses);

Eigen::MatrixXd buildRotationStateMatrix(
    const std::vector<Pose2State>& poses);

Eigen::MatrixXd recoverOptimalTranslations(
    const std::vector<Pose2State>& poses,
    const SesyncMatrices& matrices);

double evaluateSimplifiedQuadraticCost(
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& Q);

}  // namespace cora_symforce
