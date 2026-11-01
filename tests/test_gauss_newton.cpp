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


double maxPoseDifference(
    const std::vector<cora_symforce::Pose2State>& lhs,
    const std::vector<cora_symforce::Pose2State>& rhs) {

  if (lhs.size() != rhs.size()) {
    return 1e300;
  }

  double maximum = 0.0;

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    maximum = std::max(
        maximum,
        (lhs[i].R - rhs[i].R).norm());

    maximum = std::max(
        maximum,
        (lhs[i].t - rhs[i].t).norm());
  }

  return maximum;
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

  GaussNewtonOptions schur_options = qr_options;
  schur_options.linear_solver = GaussNewtonLinearSolver::kSchur;

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

  const GaussNewtonResult schur_result =
      optimizeGaussNewton(
          problem.initial_guess,
          problem.measurements,
          schur_options);

  std::cout << std::setprecision(15);

  bool passed = true;
  passed &= validateResult("SparseQR", initial_cost, qr_result);
  passed &= validateResult("Symmetric LDLT", initial_cost, ldlt_result);
  passed &= validateResult("Schur", initial_cost, schur_result);

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

  LevenbergMarquardtOptions lm_schur_options =
      lm_qr_options;

  lm_schur_options.linear_solver =
      GaussNewtonLinearSolver::kSchur;

  // Matrix-free Schur-PCG LM configuration.
  LevenbergMarquardtOptions lm_pcg_options =
      lm_qr_options;

  lm_pcg_options.linear_solver =
      GaussNewtonLinearSolver::kSchurPcg;

  lm_pcg_options.schur_pcg.max_iterations = 1000;
  lm_pcg_options.schur_pcg.relative_tolerance = 1e-12;
  lm_pcg_options.schur_pcg.absolute_tolerance = 1e-14;

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

  const GaussNewtonResult lm_schur_result =
      optimizeLevenbergMarquardt(
          problem.initial_guess,
          problem.measurements,
          lm_schur_options);

  const GaussNewtonResult lm_pcg_result =
      optimizeLevenbergMarquardt(
          problem.initial_guess,
          problem.measurements,
          lm_pcg_options);

  passed &= validateResult(
      "LM SparseQR",
      initial_cost,
      lm_qr_result);

  passed &= validateResult(
      "LM Symmetric LDLT",
      initial_cost,
      lm_ldlt_result);

  passed &= validateResult(
      "LM Schur",
      initial_cost,
      lm_schur_result);

  passed &= validateResult(
      "LM Schur PCG",
      initial_cost,
      lm_pcg_result);

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

  const double lm_schur_cost_gap =
      std::abs(
          lm_ldlt_result.final_cost -
          lm_schur_result.final_cost);

  std::cout
      << "\nFinal cost gap LM LDLT vs Schur : "
      << lm_schur_cost_gap
      << '\n';

  if (!std::isfinite(lm_schur_cost_gap) ||
      lm_schur_cost_gap > 1e-10) {
    std::cerr
        << "FAILED: LM LDLT and Schur final toy costs disagree.\n";
    passed = false;
  }

  const double lm_pcg_cost_gap =
      std::abs(
          lm_ldlt_result.final_cost -
          lm_pcg_result.final_cost);

  const double lm_pcg_pose_gap =
      maxPoseDifference(
          lm_ldlt_result.poses,
          lm_pcg_result.poses);

  std::size_t total_pcg_iterations = 0;
  double maximum_pcg_residual = 0.0;

  bool pcg_diagnostics_finite = true;

  for (const auto& iteration : lm_pcg_result.iterations) {
    total_pcg_iterations += iteration.pcg_iterations;

    if (!std::isfinite(iteration.pcg_relative_residual) ||
        iteration.pcg_relative_residual < 0.0) {
      pcg_diagnostics_finite = false;
    }

    maximum_pcg_residual = std::max(
        maximum_pcg_residual,
        iteration.pcg_relative_residual);
  }

  std::cout
      << "\nLM LDLT vs PCG final cost gap : "
      << lm_pcg_cost_gap
      << "\nLM LDLT vs PCG pose gap      : "
      << lm_pcg_pose_gap
      << "\nTotal PCG iterations         : "
      << total_pcg_iterations
      << "\nMaximum reported PCG residual: "
      << maximum_pcg_residual
      << '\n';

  if (!std::isfinite(lm_pcg_cost_gap) ||
      lm_pcg_cost_gap > 1e-10) {
    std::cerr
        << "FAILED: LM LDLT and PCG final costs disagree.\n";
    passed = false;
  }

  if (!std::isfinite(lm_pcg_pose_gap) ||
      lm_pcg_pose_gap > 1e-7) {
    std::cerr
        << "FAILED: LM LDLT and PCG final poses disagree.\n";
    passed = false;
  }

  if (total_pcg_iterations == 0) {
    std::cerr
        << "FAILED: LM PCG did not report any PCG iterations.\n";
    passed = false;
  }

  if (!pcg_diagnostics_finite ||
      maximum_pcg_residual > 1e-5) {
    std::cerr
        << "FAILED: LM PCG diagnostics are invalid.\n";
    passed = false;
  }

  const double gn_schur_cost_gap =
      std::abs(
          ldlt_result.final_cost -
          schur_result.final_cost);

  std::cout
      << "Final cost gap GN LDLT vs Schur : "
      << gn_schur_cost_gap
      << '\n';

  if (!std::isfinite(gn_schur_cost_gap) ||
      gn_schur_cost_gap > 1e-10) {
    std::cerr
        << "FAILED: GN LDLT and Schur final toy costs disagree.\n";
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

  std::cout << "\nGauss-Newton + LM QR/LDLT/Schur/Schur-PCG optimization PASSED.\n";
  return EXIT_SUCCESS;
}
