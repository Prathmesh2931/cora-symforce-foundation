#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "cora_symforce/graph_linearization.hpp"

namespace cora_symforce {

enum class SchurPcgPreconditioner {
  // P0: inverse diagonal of the damped rotation Hessian.
  kJacobiHcc,

  // P1: Schur diagonal approximation using independent
  // 2x2 translation-Hessian diagonal blocks.
  kSchurBlockJacobiDiagonal,

  // P3: Incomplete Cholesky of a sparse block-Jacobi Schur surrogate.
  kSchurIncompleteCholesky,
};

struct SchurPcgOptions {
  std::size_t max_iterations = 1000;
  double relative_tolerance = 1e-10;
  double absolute_tolerance = 1e-12;

  // Preserve the original Jacobi preconditioner as default.
  SchurPcgPreconditioner preconditioner =
      SchurPcgPreconditioner::kJacobiHcc;

  // Scale-relative positive floor for P1.
  double schur_diagonal_floor_relative = 1e-12;

  // Eigen IncompleteCholesky initial shift for P3.
  // Keep Eigen's default value for the first experiment.
  double incomplete_cholesky_initial_shift = 1e-3;
};

struct SchurPcgResult {
  // Reduced state ordering:
  // [all rotation increments, translations excluding pose-0 x/y].
  Eigen::VectorXd reduced_delta;

  std::size_t iterations = 0;
  double relative_residual = 0.0;

  // P0/P1 numerical diagnostics.
  double preconditioner_min_raw_diagonal = 0.0;
  std::size_t preconditioner_floored_diagonal_entries = 0;
};

// Solve the damped, gauge-reduced normal equations by eliminating
// translations and applying the Schur complement matrix-free:
//
//   S * dc = -gc + Hcf * (Hff + lambda I)^-1 * gf
//
//   S * v = (Hcc + lambda I) v
//          - Hcf * (Hff + lambda I)^-1 * Hfc * v
//
// The sparse translation-block factorization is reused for every PCG
// matrix-vector product. No dense Schur matrix is formed.
SchurPcgResult solveSchurPcgReducedStep(
    const GraphLinearization& linearization,
    double damping,
    const SchurPcgOptions& options = {});

}  // namespace cora_symforce
