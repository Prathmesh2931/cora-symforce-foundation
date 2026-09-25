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

  const Eigen::VectorXd translation_from_matrices =
      matrices.B1 * t + matrices.B2 * r;

  const Eigen::VectorXd rotation_from_matrices =
      matrices.B3 * r;

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

  const double residual_matrix_cost =
      translation_from_matrices.squaredNorm() +
      rotation_from_matrices.squaredNorm();

  const Eigen::MatrixXd Y =
      buildExplicitStateMatrix(problem.initial_guess);

  const double explicit_quadratic_cost =
      evaluateExplicitQuadraticCost(Y, matrices.M);

  std::cout << std::fixed << std::setprecision(12);

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
