#include <cmath>
#include <iostream>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

bool nearZero(double value, double tolerance = 1e-12) {
  return std::abs(value) <= tolerance;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  const PgoEvaluation ground_truth =
      evaluatePgo(problem.ground_truth, problem.measurements);

  if (!nearZero(ground_truth.total)) {
    std::cerr
        << "Ground-truth objective is not zero: "
        << ground_truth.total
        << '\n';

    return 1;
  }

  for (const EdgeCost& edge : ground_truth.edges) {
    if (!nearZero(edge.rotation) || !nearZero(edge.translation)) {
      std::cerr
          << "Ground-truth edge "
          << edge.i << " -> " << edge.j
          << " has non-zero residual cost\n";

      return 1;
    }
  }

  const PgoEvaluation perturbed =
      evaluatePgo(problem.initial_guess, problem.measurements);

  if (!(perturbed.total > 0.0)) {
    std::cerr << "Perturbed state should have positive cost\n";
    return 1;
  }

  std::cout << "reference PGO tests passed\n";

  return 0;
}
