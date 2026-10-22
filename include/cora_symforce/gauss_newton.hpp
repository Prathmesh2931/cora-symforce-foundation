#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

enum class GaussNewtonLinearSolver {
  kSparseQr,
  kSymmetricLdlt,
  kSchur,
};

const char* gaussNewtonLinearSolverName(GaussNewtonLinearSolver solver);

struct GaussNewtonOptions {
  std::size_t max_iterations = 30;
  double gradient_tolerance = 1e-8;
  double step_tolerance = 1e-10;
  double relative_cost_tolerance = 1e-12;
  std::size_t max_backtracking_steps = 12;
  double minimum_step_scale = 1e-6;

  // Keep SparseQR as the default/reference backend.  The symmetric backend
  // solves the gauge-reduced normal equations with SimplicialLDLT.
  GaussNewtonLinearSolver linear_solver = GaussNewtonLinearSolver::kSparseQr;

  // Optional instrumentation for visualization/validation. Disabled by default
  // so normal benchmark runs do not copy full trajectories every iteration.
  bool record_trajectory_history = false;
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

  // Trust-region / Levenberg-Marquardt diagnostics.
  // These remain zero for the original Gauss-Newton implementation.
  double damping = 0.0;
  double gain_ratio = 0.0;
  std::size_t trial_count = 0;
};

struct LevenbergMarquardtOptions {
  std::size_t max_iterations = 100;
  std::size_t max_trials_per_iteration = 12;

  double gradient_tolerance = 1e-8;
  double step_tolerance = 1e-10;
  double relative_cost_tolerance = 1e-12;

  // Damping for
  //
  //   (J_r^T J_r + lambda I) delta = -J_r^T r
  //
  // Both SparseQR and LDLT solve the same gauge-reduced damped model.
  double initial_damping = 1e-3;

  // A trial is accepted only when the nonlinear objective decreases
  // consistently with the local quadratic model.
  double acceptance_threshold = 1e-4;

  GaussNewtonLinearSolver linear_solver =
      GaussNewtonLinearSolver::kSymmetricLdlt;

  bool record_trajectory_history = false;
};

struct GaussNewtonResult {
  std::vector<Pose2State> poses;
  std::vector<GaussNewtonIteration> iterations;

  // When record_trajectory_history is enabled, entry 0 is the initial
  // trajectory and each following entry is one accepted Gauss-Newton state.
  std::vector<std::vector<Pose2State>> trajectory_history;

  double initial_cost = 0.0;
  double final_cost = 0.0;
  bool converged = false;
  std::string termination_reason;

  double total_linearization_time_ms = 0.0;
  double total_solve_time_ms = 0.0;
};

// Full pose-graph Gauss-Newton optimizer.
//
// The graph Jacobian is assembled in the global ordering
//
//   [theta_0 ... theta_(n-1),
//    t_0x t_0y ... t_(n-1)x t_(n-1)y].
//
// kSparseQr solves J * delta ~= -r directly without anchoring a pose.
//
// kSymmetricLdlt removes only pose-0 x/y translation increments, forms
// H = J_r^T J_r and g = J_r^T r, and solves H * delta_r = -g with sparse
// SimplicialLDLT.  The two removed columns select a translation reference for
// the exact global-translation nullspace observed in this formulation.
GaussNewtonResult optimizeGaussNewton(
    const std::vector<Pose2State>& initial_poses,
    const std::vector<RelativePoseMeasurement>& measurements,
    const GaussNewtonOptions& options = {});

GaussNewtonResult optimizeLevenbergMarquardt(
    const std::vector<Pose2State>& initial_poses,
    const std::vector<RelativePoseMeasurement>& measurements,
    const LevenbergMarquardtOptions& options = {});

}  // namespace cora_symforce
