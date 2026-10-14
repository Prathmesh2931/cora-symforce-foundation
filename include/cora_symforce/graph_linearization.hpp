#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct GraphLinearization {
  Eigen::VectorXd residual;

  // Constrained variables: one SO(2) angle per pose.
  Eigen::SparseMatrix<double> Jc;

  // Free variables: two Euclidean translation coordinates per pose.
  Eigen::SparseMatrix<double> Jf;

  Eigen::SparseMatrix<double> Hcc;
  Eigen::SparseMatrix<double> Hcf;
  Eigen::SparseMatrix<double> Hff;

  double cost() const {
    return residual.squaredNorm();
  }
};

GraphLinearization linearizeGraph(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements);

}  // namespace cora_symforce
