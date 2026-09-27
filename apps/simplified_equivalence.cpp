#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/sesync_matrices.hpp"
#include "cora_symforce/simplified_pgo.hpp"
#include "cora_symforce/toy_problem.hpp"

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  const SesyncMatrices matrices =
      buildSesyncMatrices(
          problem.measurements,
          problem.initial_guess.size());

  const SimplifiedProblem simplified =
      buildSimplifiedProblem(
          problem.measurements,
          problem.initial_guess.size());

  // Keep the current rotations fixed and solve only for t*(R).
  const Eigen::MatrixXd optimal_translations =
      recoverOptimalTranslations(
          problem.initial_guess,
          matrices);

  std::vector<Pose2State> recovered_state =
      problem.initial_guess;

  for (std::size_t i = 0;
       i < recovered_state.size();
       ++i) {
    recovered_state[i].t =
        optimal_translations.col(i);
  }

  // Cost at the original perturbed state.
  const double original_cost =
      evaluatePgo(
          problem.initial_guess,
          problem.measurements)
          .total;

  // Explicit PGO cost after replacing translations with t*(R).
  const double explicit_recovered_cost =
      evaluatePgo(
          recovered_state,
          problem.measurements)
          .total;

  const Eigen::MatrixXd R =
      buildRotationStateMatrix(
          problem.initial_guess);

  // Reduced rotation-only objective: trace(R * Q * R^T).
  const double simplified_cost =
      evaluateSimplifiedQuadraticCost(
          R,
          simplified.Q);

  std::cout << std::fixed << std::setprecision(12);

  std::cout << "Recovered translations\n";
  std::cout << optimal_translations << "\n\n";

  std::cout
      << "original explicit cost:          "
      << original_cost
      << '\n';

  std::cout
      << "explicit cost at t*(R):          "
      << explicit_recovered_cost
      << '\n';

  std::cout
      << "simplified Q cost:               "
      << simplified_cost
      << '\n';

  // Main equivalence check:
  // F(R, t*(R)) should match trace(R * Q * R^T).
  std::cout
      << "explicit vs simplified diff:     "
      << std::abs(
             explicit_recovered_cost -
             simplified_cost)
      << '\n';

  std::cout
      << "Q symmetry error:                "
      << (simplified.Q -
          simplified.Q.transpose())
             .norm()
      << '\n';

  // Pi should behave as an orthogonal projector:
  // Pi^T = Pi and Pi^2 = Pi.
  std::cout
      << "Pi symmetry error:               "
      << (simplified.Pi -
          simplified.Pi.transpose())
             .norm()
      << '\n';

  std::cout
      << "Pi idempotence error:            "
      << (simplified.Pi * simplified.Pi -
          simplified.Pi)
             .norm()
      << '\n';

  return 0;
}
