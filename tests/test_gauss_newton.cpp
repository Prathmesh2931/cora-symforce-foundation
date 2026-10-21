#include <algorithm>
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

bool validateResult(
    const char* label,
    const double initial_cost,
    const cora_symforce::GaussNewtonResult& result) {
  using namespace cora_symforce;

  const double final_cost = result.final_cost;

  std::cout << "\n" << label << '\n';
  std::cout << "Initial cost       : " << initial_cost << '\n';
  std::cout << "Final cost         : " << final_cost << '\n';
  std::cout << "Iterations         : " << result.iterations.size() << '\n';
  std::cout << "Converged          : "
            << (result.converged ? "yes" : "no") << '\n';
  std::cout << "Termination        : "
            << result.termination_reason << '\n';
  std::cout << "Solve total [ms]   : "
            << result.total_solve_time_ms << '\n';

  if (!result.iterations.empty()) {
    const GaussNewtonIteration& last = result.iterations.back();
    std::cout << "Last rank/dim      : " << last.numerical_rank << '\n';
    std::cout << "Last gradient norm : " << last.gradient_norm << '\n';
    std::cout << "Last step norm     : " << last.step_norm << '\n';
  }

  bool passed = true;

  if (!(final_cost < initial_cost)) {
    std::cerr << "FAILED " << label << ": objective did not decrease.\n";
    passed = false;
  }

  if (!(final_cost < 1e-10)) {
    std::cerr << "FAILED " << label
              << ": optimizer did not recover zero-residual toy graph.\n";
    passed = false;
  }

  if (!finitePoses(result.poses)) {
    std::cerr << "FAILED " << label << ": optimized poses contain NaN/Inf.\n";
    passed = false;
  }

  if (!result.converged) {
    std::cerr << "FAILED " << label
              << ": optimizer did not report convergence.\n";
    passed = false;
  }

  if (result.trajectory_history.empty()) {
    std::cerr << "FAILED " << label
              << ": trajectory history was not recorded.\n";
    passed = false;
  } else {
    if (result.trajectory_history.size() != result.iterations.size() + 1) {
      std::cerr << "FAILED " << label
                << ": trajectory history length is inconsistent.\n";
      passed = false;
    }

    const auto& recorded_final = result.trajectory_history.back();
    if (recorded_final.size() != result.poses.size()) {
      std::cerr << "FAILED " << label
                << ": recorded final trajectory size is inconsistent.\n";
      passed = false;
    } else {
      double max_difference = 0.0;
      for (std::size_t i = 0; i < result.poses.size(); ++i) {
        max_difference = std::max(
            max_difference,
            (recorded_final[i].R - result.poses[i].R).norm());
        max_difference = std::max(
            max_difference,
            (recorded_final[i].t - result.poses[i].t).norm());
      }
      if (max_difference > 1e-12) {
        std::cerr << "FAILED " << label
                  << ": recorded final trajectory does not match result.\n";
        passed = false;
      }
    }
  }

  return passed;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();

  const double initial_cost =
      evaluateSymForcePgo(problem.initial_guess, problem.measurements).total;

  GaussNewtonOptions qr_options;
  qr_options.max_iterations = 30;
  qr_options.gradient_tolerance = 1e-10;
  qr_options.step_tolerance = 1e-12;
  qr_options.relative_cost_tolerance = 1e-14;
  qr_options.linear_solver = GaussNewtonLinearSolver::kSparseQr;
  qr_options.record_trajectory_history = true;

  GaussNewtonOptions ldlt_options = qr_options;
  ldlt_options.linear_solver = GaussNewtonLinearSolver::kSymmetricLdlt;

  const GaussNewtonResult qr_result =
      optimizeGaussNewton(
          problem.initial_guess,
          problem.measurements,
          qr_options);

  const GaussNewtonResult ldlt_result =
      optimizeGaussNewton(
          problem.initial_guess,
          problem.measurements,
          ldlt_options);

  std::cout << std::setprecision(15);

  bool passed = true;
  passed &= validateResult("SparseQR", initial_cost, qr_result);
  passed &= validateResult("Symmetric LDLT", initial_cost, ldlt_result);

  LevenbergMarquardtOptions lm_qr_options;
  lm_qr_options.max_iterations = 50;
  lm_qr_options.max_trials_per_iteration = 12;
  lm_qr_options.gradient_tolerance = 1e-10;
  lm_qr_options.step_tolerance = 1e-12;
  lm_qr_options.relative_cost_tolerance = 1e-14;
  lm_qr_options.initial_damping = 1e-3;
  lm_qr_options.acceptance_threshold = 1e-4;
  lm_qr_options.linear_solver =
      GaussNewtonLinearSolver::kSparseQr;
  lm_qr_options.record_trajectory_history = true;

  LevenbergMarquardtOptions lm_ldlt_options =
      lm_qr_options;

  lm_ldlt_options.linear_solver =
      GaussNewtonLinearSolver::kSymmetricLdlt;

  const GaussNewtonResult lm_qr_result =
      optimizeLevenbergMarquardt(
          problem.initial_guess,
          problem.measurements,
          lm_qr_options);

  const GaussNewtonResult lm_ldlt_result =
      optimizeLevenbergMarquardt(
          problem.initial_guess,
          problem.measurements,
          lm_ldlt_options);

  passed &= validateResult(
      "LM SparseQR",
      initial_cost,
      lm_qr_result);

  passed &= validateResult(
      "LM Symmetric LDLT",
      initial_cost,
      lm_ldlt_result);

  const double lm_final_cost_gap =
      std::abs(
          lm_qr_result.final_cost -
          lm_ldlt_result.final_cost);

  std::cout
      << "\nFinal cost gap LM QR vs LDLT : "
      << lm_final_cost_gap
      << '\n';

  if (!std::isfinite(lm_final_cost_gap) ||
      lm_final_cost_gap > 1e-10) {
    std::cerr
        << "FAILED: LM QR and LDLT final toy costs disagree.\n";
    passed = false;
  }

  const double final_cost_gap =
      std::abs(qr_result.final_cost - ldlt_result.final_cost);
  std::cout << "\nFinal cost gap QR vs LDLT : " << final_cost_gap << '\n';

  if (!std::isfinite(final_cost_gap) || final_cost_gap > 1e-10) {
    std::cerr << "FAILED: QR and LDLT final toy costs disagree.\n";
    passed = false;
  }

  if (!passed) {
    return EXIT_FAILURE;
  }

  std::cout << "\nGauss-Newton + LM QR/LDLT optimization PASSED.\n";
  return EXIT_SUCCESS;
}
