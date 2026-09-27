#include <cmath>
#include <iomanip>
#include <iostream>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/sesync_matrices.hpp"
#include "cora_symforce/toy_problem.hpp"

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  const SesyncMatrices matrices =
      buildSesyncMatrices(
          problem.measurements,
          problem.initial_guess.size());

  const Eigen::VectorXd t =
      stackTranslations(problem.initial_guess);

  const Eigen::VectorXd r =
      stackRotations(problem.initial_guess);

  // Matrix-form translation residual:
  // B1 * t gives t_j - t_i
  // B2 * r gives -R_i * t_ij
  const Eigen::VectorXd translation_from_matrices =
      matrices.B1 * t + matrices.B2 * r;

  // Matrix-form rotation residual:
  // B3 * r gives vec(R_j - R_i * R_ij).
  const Eigen::VectorXd rotation_from_matrices =
      matrices.B3 * r;

  // Direct residuals are kept as the reference implementation.
  // These should match the B1/B2/B3 representation numerically.
  const Eigen::VectorXd translation_direct =
      weightedTranslationResiduals(
          problem.initial_guess,
          problem.measurements);

  const Eigen::VectorXd rotation_direct =
      weightedRotationResiduals(
          problem.initial_guess,
          problem.measurements);

  const PgoEvaluation direct_evaluation =
      evaluatePgo(
          problem.initial_guess,
          problem.measurements);

  // Same PGO objective, evaluated from the stacked residuals.
  const double residual_matrix_cost =
      translation_from_matrices.squaredNorm() +
      rotation_from_matrices.squaredNorm();

  // Explicit SE-Sync state:
  // Y = [t_0 ... t_n | R_0 ... R_n]
  const Eigen::MatrixXd Y =
      buildExplicitStateMatrix(problem.initial_guess);

  // Same objective again, now written as trace(Y * M * Y^T).
  const double explicit_quadratic_cost =
      evaluateExplicitQuadraticCost(Y, matrices.M);

  std::cout << std::fixed << std::setprecision(12);

  // Residual-level checks are stronger than comparing only total costs.
  std::cout
      << "translation residual difference: "
      << (translation_from_matrices - translation_direct).norm()
      << '\n';

  std::cout
      << "rotation residual difference:    "
      << (rotation_from_matrices - rotation_direct).norm()
      << '\n';

  std::cout
      << "direct PGO cost:                 "
      << direct_evaluation.total
      << '\n';

  std::cout
      << "B1/B2/B3 cost:                   "
      << residual_matrix_cost
      << '\n';

  std::cout
      << "explicit M cost:                 "
      << explicit_quadratic_cost
      << '\n';

  // Main objective check:
  // direct PGO == B1/B2/B3 form == explicit M form.
  std::cout
      << "direct vs M difference:          "
      << std::abs(
             direct_evaluation.total -
             explicit_quadratic_cost)
      << '\n';

  std::cout
      << "M symmetry error:                "
      << (matrices.M - matrices.M.transpose()).norm()
      << '\n';

  return 0;
}
