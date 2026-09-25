#include <cmath>
#include <iostream>
#include <vector>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/sesync_matrices.hpp"
#include "cora_symforce/simplified_pgo.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

constexpr double kTolerance = 1e-10;

bool checkState(
    const std::vector<cora_symforce::Pose2State>& poses,
    const std::vector<cora_symforce::RelativePoseMeasurement>& measurements) {
  using namespace cora_symforce;

  const SesyncMatrices matrices =
      buildSesyncMatrices(
          measurements,
          poses.size());

  const SimplifiedProblem simplified =
      buildSimplifiedProblem(
          measurements,
          poses.size());

  const Eigen::MatrixXd translations =
      recoverOptimalTranslations(
          poses,
          matrices);

  std::vector<Pose2State> recovered_state =
      poses;

  for (std::size_t i = 0;
       i < recovered_state.size();
       ++i) {
    recovered_state[i].t =
        translations.col(i);
  }

  const double explicit_cost =
      evaluatePgo(
          recovered_state,
          measurements)
          .total;

  const Eigen::MatrixXd R =
      buildRotationStateMatrix(poses);

  const double simplified_cost =
      evaluateSimplifiedQuadraticCost(
          R,
          simplified.Q);

  if (std::abs(explicit_cost - simplified_cost) >
      kTolerance) {
    std::cerr
        << "explicit/simplified mismatch: "
        << std::abs(explicit_cost - simplified_cost)
        << '\n';

    return false;
  }

  if ((simplified.Q -
       simplified.Q.transpose())
          .norm() > kTolerance) {
    std::cerr << "Q is not symmetric\n";
    return false;
  }

  if ((simplified.Pi -
       simplified.Pi.transpose())
          .norm() > kTolerance) {
    std::cerr << "Pi is not symmetric\n";
    return false;
  }

  if ((simplified.Pi * simplified.Pi -
       simplified.Pi)
          .norm() > kTolerance) {
    std::cerr << "Pi is not idempotent\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem =
      makeToyProblem();

  if (!checkState(
          problem.ground_truth,
          problem.measurements)) {
    std::cerr
        << "ground-truth rotations failed\n";

    return 1;
  }

  if (!checkState(
          problem.initial_guess,
          problem.measurements)) {
    std::cerr
        << "perturbed rotations failed\n";

    return 1;
  }

  std::cout
      << "explicit/simplified equivalence tests passed\n";

  return 0;
}
