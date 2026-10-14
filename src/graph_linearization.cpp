#include "cora_symforce/graph_linearization.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "cora_symforce/symforce_pgo_cost.hpp"
#include "edge_residual.h"

namespace cora_symforce {
namespace {

double rotationAngle(const Eigen::Matrix2d& R) {
  return std::atan2(R(1, 0), R(0, 0));
}

void validateMeasurementIndices(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement) {
  if (measurement.i >= poses.size() || measurement.j >= poses.size()) {
    throw std::out_of_range(
        "RelativePoseMeasurement references an invalid pose index.");
  }
}

void addIfNonzero(
    std::vector<Eigen::Triplet<double>>& triplets,
    Eigen::Index row,
    Eigen::Index col,
    double value) {
  if (value != 0.0) {
    triplets.emplace_back(row, col, value);
  }
}

}  // namespace

GraphLinearization linearizeGraph(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements) {
  const Eigen::Index n = static_cast<Eigen::Index>(poses.size());
  const Eigen::Index m = static_cast<Eigen::Index>(measurements.size());
  const Eigen::Index residual_rows = 6 * m;

  GraphLinearization result;
  result.residual = Eigen::VectorXd::Zero(residual_rows);
  result.Jc.resize(residual_rows, n);
  result.Jf.resize(residual_rows, 2 * n);

  std::vector<Eigen::Triplet<double>> constrained_triplets;
  std::vector<Eigen::Triplet<double>> free_triplets;
  constrained_triplets.reserve(static_cast<std::size_t>(12 * m));
  free_triplets.reserve(static_cast<std::size_t>(12 * m));

  for (Eigen::Index edge_index = 0; edge_index < m; ++edge_index) {
    const RelativePoseMeasurement& measurement =
        measurements[static_cast<std::size_t>(edge_index)];
    validateMeasurementIndices(poses, measurement);

    const Pose2State& pose_i = poses[measurement.i];
    const Pose2State& pose_j = poses[measurement.j];

    const double theta_i = rotationAngle(pose_i.R);
    const double theta_j = rotationAngle(pose_j.R);

    const Eigen::Matrix<double, 6, 1> edge_residual =
        cora_symforce_generated::EdgeResidual<double>(
            theta_i,
            pose_i.t,
            theta_j,
            pose_j.t,
            measurement.R,
            measurement.t,
            measurement.kappa,
            measurement.tau);

    const SymForceEdgeJacobian edge_jacobian =
        evaluateSymForceEdgeJacobian(poses, measurement);

    const Eigen::Index row_offset = 6 * edge_index;
    result.residual.segment<6>(row_offset) = edge_residual;

    const Eigen::Index i = static_cast<Eigen::Index>(measurement.i);
    const Eigen::Index j = static_cast<Eigen::Index>(measurement.j);

    // Local state ordering:
    // [theta_i, t_ix, t_iy, theta_j, t_jx, t_jy].
    for (Eigen::Index row = 0; row < 6; ++row) {
      const Eigen::Index global_row = row_offset + row;

      // Constrained SO(2) variables.
      addIfNonzero(constrained_triplets, global_row, i,
                   edge_jacobian(row, 0));
      addIfNonzero(constrained_triplets, global_row, j,
                   edge_jacobian(row, 3));

      // Free translation variables.
      addIfNonzero(free_triplets, global_row, 2 * i,
                   edge_jacobian(row, 1));
      addIfNonzero(free_triplets, global_row, 2 * i + 1,
                   edge_jacobian(row, 2));
      addIfNonzero(free_triplets, global_row, 2 * j,
                   edge_jacobian(row, 4));
      addIfNonzero(free_triplets, global_row, 2 * j + 1,
                   edge_jacobian(row, 5));
    }
  }

  result.Jc.setFromTriplets(
      constrained_triplets.begin(), constrained_triplets.end());
  result.Jf.setFromTriplets(
      free_triplets.begin(), free_triplets.end());
  result.Jc.makeCompressed();
  result.Jf.makeCompressed();

  result.Hcc = result.Jc.transpose() * result.Jc;
  result.Hcf = result.Jc.transpose() * result.Jf;
  result.Hff = result.Jf.transpose() * result.Jf;
  result.Hcc.makeCompressed();
  result.Hcf.makeCompressed();
  result.Hff.makeCompressed();

  return result;
}

}  // namespace cora_symforce
