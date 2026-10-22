#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "cora_symforce/graph_linearization.hpp"

namespace cora_symforce {

struct LinearSolverComparisonOptions {
  // Repeat the fixed-linearization solves so the reported timings are medians
  // rather than a single noisy measurement.
  std::size_t repetitions = 3;
};

struct LinearSolverComparison {
  Eigen::Index residual_rows = 0;
  Eigen::Index state_columns = 0;
  Eigen::Index reduced_state_columns = 0;

  Eigen::Index jacobian_nonzeros = 0;
  Eigen::Index normal_nonzeros = 0;
  Eigen::Index qr_rank = 0;

  // We remove only the two exact global-translation gauge directions by
  // fixing the first pose's x/y increments to zero in the LDLT system.
  Eigen::Index fixed_tx_column = -1;
  Eigen::Index fixed_ty_column = -1;

  double normal_symmetry_error = 0.0;

  double qr_factorization_time_ms = 0.0;
  double qr_solve_time_ms = 0.0;

  double normal_formation_time_ms = 0.0;
  double ldlt_factorization_time_ms = 0.0;
  double ldlt_solve_time_ms = 0.0;

  double qr_linearized_residual_norm = 0.0;
  double ldlt_linearized_residual_norm = 0.0;
  double linearized_residual_norm_gap = 0.0;

  // Gauge-equivalent steps may differ numerically, so compare their action in
  // residual space rather than comparing delta vectors directly.
  double prediction_difference_norm = 0.0;

  // Relative residual of H * delta + g = 0 for the reduced symmetric system.
  double ldlt_relative_normal_residual = 0.0;

  // Explicit Schur-complement reference solve.  Rotations are retained,
  // while pose-0 x/y translations are fixed and the remaining translations
  // are eliminated through Hff.
  Eigen::Index schur_rotation_columns = 0;
  Eigen::Index schur_translation_columns = 0;

  double schur_symmetry_error = 0.0;
  double schur_linearized_residual_norm = 0.0;

  // Compare Schur and full LDLT in residual space because gauge-equivalent
  // state increments need not be compared component-wise.
  double schur_prediction_difference_norm = 0.0;

  // Relative residual of the same gauge-reduced full normal equations,
  // evaluated using the Schur-complement solution.
  double schur_relative_normal_residual = 0.0;
};

// Compare two linear solvers on exactly one already-computed graph
// linearization:
//
//   1. SparseQR solves       J * delta ~= -r
//   2. SimplicialLDLT solves (J_r^T J_r) delta_r = -J_r^T r
//
// J_r removes only pose-0 translation increments, which selects a numerical
// reference frame for the two exact global-translation null directions.  The
// nonlinear optimizer is not modified by this experiment.
LinearSolverComparison compareLinearSolvers(
    const GraphLinearization& linearization,
    const LinearSolverComparisonOptions& options = {});

}  // namespace cora_symforce
