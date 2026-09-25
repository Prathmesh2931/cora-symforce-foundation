#include <cmath>
#include <iostream>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/sesync_matrices.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

constexpr double kTolerance = 1e-12;

bool vectorsNear(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs) {
  return (lhs - rhs).norm() <= kTolerance;
}

bool scalarsNear(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= kTolerance;
}

bool checkState(
    const std::vector<cora_symforce::Pose2State>& poses,
    const std::vector<cora_symforce::RelativePoseMeasurement>& measurements) {
  using namespace cora_symforce;

  const SesyncMatrices matrices =
      buildSesyncMatrices(measurements, poses.size());

  const Eigen::VectorXd t = stackTranslations(poses);
  const Eigen::VectorXd r = stackRotations(poses);

  const Eigen::VectorXd translation_matrix =
      matrices.B1 * t + matrices.B2 * r;

  const Eigen::VectorXd rotation_matrix =
      matrices.B3 * r;

  const Eigen::VectorXd translation_direct =
      weightedTranslationResiduals(
          poses,
          measurements);

  const Eigen::VectorXd rotation_direct =
      weightedRotationResiduals(
          poses,
          measurements);

  if (!vectorsNear(translation_matrix, translation_direct)) {
    std::cerr
        << "translation residual mismatch: "
        << (translation_matrix - translation_direct).norm()
        << '\n';

    return false;
  }

  if (!vectorsNear(rotation_matrix, rotation_direct)) {
    std::cerr
        << "rotation residual mismatch: "
        << (rotation_matrix - rotation_direct).norm()
        << '\n';

    return false;
  }

  const double residual_matrix_cost =
      translation_matrix.squaredNorm() +
      rotation_matrix.squaredNorm();

  const double direct_cost =
      evaluatePgo(poses, measurements).total;

  if (!scalarsNear(residual_matrix_cost, direct_cost)) {
    std::cerr
        << "B1/B2/B3 objective mismatch: "
        << std::abs(residual_matrix_cost - direct_cost)
        << '\n';

    return false;
  }

  const Eigen::MatrixXd Y =
      buildExplicitStateMatrix(poses);

  const double explicit_quadratic_cost =
      evaluateExplicitQuadraticCost(
          Y,
          matrices.M);

  if (!scalarsNear(explicit_quadratic_cost, direct_cost)) {
    std::cerr
        << "explicit M objective mismatch: "
        << std::abs(explicit_quadratic_cost - direct_cost)
        << '\n';

    return false;
  }

  if ((matrices.M - matrices.M.transpose()).norm() >
      kTolerance) {
    std::cerr << "explicit data matrix M is not symmetric\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  if (!checkState(
          problem.ground_truth,
          problem.measurements)) {
    std::cerr << "ground-truth state failed\n";
    return 1;
  }

  if (!checkState(
          problem.initial_guess,
          problem.measurements)) {
    std::cerr << "perturbed state failed\n";
    return 1;
  }

  std::cout
      << "SE-Sync explicit quadratic-form tests passed\n";

  return 0;
}
