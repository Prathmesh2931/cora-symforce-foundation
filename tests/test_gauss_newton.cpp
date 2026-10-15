#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cora_symforce/gauss_newton.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

bool finitePoses(const std::vector<cora_symforce::Pose2State>& poses) {
  for (const auto& pose : poses) {
    if (!pose.R.array().isFinite().all() ||
        !pose.t.array().isFinite().all()) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  const double initial_cost =
      evaluateSymForcePgo(
          problem.initial_guess,
          problem.measurements)
          .total;

  GaussNewtonOptions options;
  options.max_iterations = 30;
  options.gradient_tolerance = 1e-10;
  options.step_tolerance = 1e-12;
  options.relative_cost_tolerance = 1e-14;

  const GaussNewtonResult result =
      optimizeGaussNewton(
          problem.initial_guess,
          problem.measurements,
          options);

  const double final_cost =
      evaluateSymForcePgo(
          result.poses,
          problem.measurements)
          .total;

  std::cout << std::setprecision(15);
  std::cout << "Initial cost       : " << initial_cost << '\n';
  std::cout << "Final cost         : " << final_cost << '\n';
  std::cout << "Iterations         : " << result.iterations.size() << '\n';
  std::cout << "Converged          : "
            << (result.converged ? "yes" : "no") << '\n';
  std::cout << "Termination        : "
            << result.termination_reason << '\n';

  if (!result.iterations.empty()) {
    const GaussNewtonIteration& last = result.iterations.back();
    std::cout << "Last numerical rank: " << last.numerical_rank << '\n';
    std::cout << "Last gradient norm : " << last.gradient_norm << '\n';
    std::cout << "Last step norm     : " << last.step_norm << '\n';
  }

  bool passed = true;

  if (!(final_cost < initial_cost)) {
    std::cerr << "FAILED: objective did not decrease.\n";
    passed = false;
  }

  if (!(final_cost < 1e-10)) {
    std::cerr
        << "FAILED: optimizer did not recover the zero-residual toy graph.\n";
    passed = false;
  }

  if (!finitePoses(result.poses)) {
    std::cerr << "FAILED: optimized poses contain NaN/Inf.\n";
    passed = false;
  }

  if (!result.converged) {
    std::cerr << "FAILED: optimizer did not report convergence.\n";
    passed = false;
  }

  if (!passed) {
    return EXIT_FAILURE;
  }

  std::cout << "\nGauss-Newton synthetic optimization PASSED.\n";
  return EXIT_SUCCESS;
}
