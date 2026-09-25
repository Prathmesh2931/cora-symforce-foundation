#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

void printEvaluation(
    const std::string& name,
    const cora_symforce::PgoEvaluation& evaluation) {
  std::cout << '\n' << name << '\n';
  std::cout << std::string(name.size(), '-') << '\n';

  for (const auto& edge : evaluation.edges) {
    std::cout
        << edge.i << " -> " << edge.j
        << "  rotation=" << edge.rotation
        << "  translation=" << edge.translation
        << "  total=" << edge.total()
        << '\n';
  }

  std::cout << "total=" << evaluation.total << '\n';
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  std::cout << std::fixed << std::setprecision(10);

  printEvaluation(
      "Ground truth",
      evaluatePgo(problem.ground_truth, problem.measurements));

  printEvaluation(
      "Perturbed state",
      evaluatePgo(problem.initial_guess, problem.measurements));

  return 0;
}
