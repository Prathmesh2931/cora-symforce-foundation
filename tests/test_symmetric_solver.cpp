#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/symmetric_solver.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

bool checkNear(
    const double value,
    const double tolerance,
    const char* label) {
  std::cout
      << std::left
      << std::setw(44)
      << label
      << " = "
      << std::scientific
      << value
      << '\n';

  if (!std::isfinite(value) || value > tolerance) {
    std::cerr
        << "FAILED: " << label
        << " exceeded tolerance " << tolerance << '\n';
    return false;
  }

  return true;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();
  const GraphLinearization linearization =
      linearizeGraph(problem.initial_guess, problem.measurements);

  LinearSolverComparisonOptions options;
  options.repetitions = 1;

  const LinearSolverComparison comparison =
      compareLinearSolvers(linearization, options);

  std::cout << std::setprecision(12);
  std::cout << "Residual rows                 : "
            << comparison.residual_rows << '\n';
  std::cout << "Full state columns            : "
            << comparison.state_columns << '\n';
  std::cout << "Gauge-reduced columns         : "
            << comparison.reduced_state_columns << '\n';
  std::cout << "SparseQR numerical rank       : "
            << comparison.qr_rank << '\n';
  std::cout << "J nonzeros                    : "
            << comparison.jacobian_nonzeros << '\n';
  std::cout << "H nonzeros                    : "
            << comparison.normal_nonzeros << '\n';

  std::cout << "Schur rotation columns        : "
            << comparison.schur_rotation_columns << '\n';

  std::cout << "Schur translation columns     : "
            << comparison.schur_translation_columns << "\n\n";

  bool passed = true;

  passed &= checkNear(
      comparison.normal_symmetry_error,
      1e-12,
      "H symmetry max error");

  passed &= checkNear(
      comparison.linearized_residual_norm_gap,
      1e-9,
      "QR vs LDLT residual-norm gap");

  passed &= checkNear(
      comparison.prediction_difference_norm,
      1e-8,
      "QR vs LDLT prediction difference");

  passed &= checkNear(
      comparison.ldlt_relative_normal_residual,
      1e-9,
      "LDLT relative normal residual");

  passed &= checkNear(
      comparison.schur_symmetry_error,
      1e-10,
      "Schur symmetry max error");

  passed &= checkNear(
      std::abs(
          comparison.schur_linearized_residual_norm -
          comparison.ldlt_linearized_residual_norm),
      1e-9,
      "Schur vs LDLT residual-norm gap");

  passed &= checkNear(
      comparison.schur_prediction_difference_norm,
      1e-8,
      "LDLT vs Schur prediction difference");

  passed &= checkNear(
      comparison.schur_relative_normal_residual,
      1e-9,
      "Schur relative normal residual");

  if (comparison.reduced_state_columns !=
      comparison.state_columns - 2) {
    std::cerr
        << "FAILED: symmetric system did not remove exactly two "
           "translation gauge columns.\n";
    passed = false;
  }

  if (comparison.schur_rotation_columns !=
      linearization.Jc.cols()) {
    std::cerr
        << "FAILED: Schur rotation dimension does not match Jc.\n";
    passed = false;
  }

  if (comparison.schur_translation_columns !=
      linearization.Jf.cols() - 2) {
    std::cerr
        << "FAILED: Schur translation dimension did not remove exactly "
           "the pose-0 x/y gauge coordinates.\n";
    passed = false;
  }

  if (comparison.schur_rotation_columns +
          comparison.schur_translation_columns !=
      comparison.reduced_state_columns) {
    std::cerr
        << "FAILED: Schur block dimensions do not reconstruct the "
           "gauge-reduced state dimension.\n";
    passed = false;
  }

  if (!(comparison.qr_factorization_time_ms >= 0.0) ||
      !(comparison.qr_solve_time_ms >= 0.0) ||
      !(comparison.normal_formation_time_ms >= 0.0) ||
      !(comparison.ldlt_factorization_time_ms >= 0.0) ||
      !(comparison.ldlt_solve_time_ms >= 0.0)) {
    std::cerr << "FAILED: solver timings are not finite/non-negative.\n";
    passed = false;
  }

  if (!passed) {
    return EXIT_FAILURE;
  }

  std::cout << "\nSparseQR / LDLT / Schur checks PASSED.\n";
  return EXIT_SUCCESS;
}
