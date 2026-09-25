#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct SesyncMatrices {
  Eigen::MatrixXd B1;
  Eigen::MatrixXd B2;
  Eigen::MatrixXd B3;
  Eigen::MatrixXd M;
};

SesyncMatrices buildSesyncMatrices(
    const std::vector<RelativePoseMeasurement>& measurements,
    std::size_t num_poses);

Eigen::VectorXd stackTranslations(
    const std::vector<Pose2State>& poses);

Eigen::VectorXd stackRotations(
    const std::vector<Pose2State>& poses);

Eigen::MatrixXd buildExplicitStateMatrix(
    const std::vector<Pose2State>& poses);

Eigen::VectorXd weightedTranslationResiduals(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements);

Eigen::VectorXd weightedRotationResiduals(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements);

double evaluateExplicitQuadraticCost(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& M);

}  // namespace cora_symforce
