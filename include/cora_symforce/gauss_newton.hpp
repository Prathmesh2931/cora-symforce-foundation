#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct GaussNewtonOptions {
  std::size_t max_iterations = 30;
  double gradient_tolerance = 1e-8;
  double step_tolerance = 1e-10;
  double relative_cost_tolerance = 1e-12;
  std::size_t max_backtracking_steps = 12;
  double minimum_step_scale = 1e-6;
};

struct GaussNewtonIteration {
  std::size_t iteration = 0;
  double cost_before = 0.0;
  double cost_after = 0.0;
  double gradient_norm = 0.0;
  double step_norm = 0.0;
  double step_scale = 0.0;
  Eigen::Index numerical_rank = 0;
  double linearization_time_ms = 0.0;
  double solve_time_ms = 0.0;
};

struct GaussNewtonResult {
  std::vector<Pose2State> poses;
  std::vector<GaussNewtonIteration> iterations;

  double initial_cost = 0.0;
  double final_cost = 0.0;
  bool converged = false;
  std::string termination_reason;

  double total_linearization_time_ms = 0.0;
  double total_solve_time_ms = 0.0;
};

// Full pose-graph Gauss-Newton baseline.
//
// The graph Jacobian is assembled in the global ordering
//
//   [theta_0 ... theta_(n-1),
//    t_0x t_0y ... t_(n-1)x t_(n-1)y].
//
// We solve J * delta ~= -r directly with sparse rank-revealing QR instead of
// forming/solving the gauge-singular normal equations. No pose is anchored by
// this baseline solver.
GaussNewtonResult optimizeGaussNewton(
    const std::vector<Pose2State>& initial_poses,
    const std::vector<RelativePoseMeasurement>& measurements,
    const GaussNewtonOptions& options = {});

}  // namespace cora_symforce
