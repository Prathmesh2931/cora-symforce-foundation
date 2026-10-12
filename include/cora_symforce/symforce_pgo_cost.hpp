#pragma once

#include <Eigen/Core>

#include <vector>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

using SymForceEdgeJacobian =
    Eigen::Matrix<double, 6, 6>;


/**
 * Evaluate one SE-Sync relative-pose edge using the
 * SymForce-generated residual kernel.
 *
 * Residual ordering:
 *
 *   [ vec(R_j - R_i R_ij);
 *     t_j - t_i - R_i t_ij ]
 *
 * with square-root precision weighting already applied.
 */
EdgeCost evaluateSymForceEdge(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement);


/**
 * Evaluate the complete PGO cost using the generated
 * SymForce edge residual.
 */
PgoEvaluation evaluateSymForcePgo(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements);


/**
 * Evaluate the generated 6x6 edge Jacobian with respect to
 *
 *   [theta_i, t_ix, t_iy, theta_j, t_jx, t_jy].
 */
SymForceEdgeJacobian evaluateSymForceEdgeJacobian(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement);

}  // namespace cora_symforce
