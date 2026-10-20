#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cora_symforce/gauss_newton.hpp"
#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/trajectory_io.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct RunSummary {
  double initial_cost = 0.0;
  double final_cost = 0.0;
  std::size_t iterations = 0;
  bool converged = false;
  std::string termination_reason;
  double linearization_ms = 0.0;
  double solve_ms = 0.0;
  double wall_ms = 0.0;
};

double milliseconds(
    const Clock::time_point& start,
    const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

std::vector<RunSummary> runSolver(
    const std::vector<cora_symforce::Pose2State>& initial_poses,
    const std::vector<cora_symforce::RelativePoseMeasurement>& measurements,
    const cora_symforce::GaussNewtonOptions& options,
    const std::size_t repetitions) {
  std::vector<RunSummary> runs;
  runs.reserve(repetitions);

  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    const auto start = Clock::now();
    const cora_symforce::GaussNewtonResult result =
        cora_symforce::optimizeGaussNewton(
            initial_poses,
            measurements,
            options);
    const auto end = Clock::now();

    RunSummary summary;
    summary.initial_cost = result.initial_cost;
    summary.final_cost = result.final_cost;
    summary.iterations = result.iterations.size();
    summary.converged = result.converged;
    summary.termination_reason = result.termination_reason;
    summary.linearization_ms = result.total_linearization_time_ms;
    summary.solve_ms = result.total_solve_time_ms;
    summary.wall_ms = milliseconds(start, end);
    runs.push_back(summary);
  }

  return runs;
}

void printSolverSummary(
    const char* label,
    const std::vector<RunSummary>& runs) {
  if (runs.empty()) {
    return;
  }

  std::vector<double> linearization;
  std::vector<double> solve;
  std::vector<double> wall;
  std::vector<double> final_costs;

  linearization.reserve(runs.size());
  solve.reserve(runs.size());
  wall.reserve(runs.size());
  final_costs.reserve(runs.size());

  for (const RunSummary& run : runs) {
    linearization.push_back(run.linearization_ms);
    solve.push_back(run.solve_ms);
    wall.push_back(run.wall_ms);
    final_costs.push_back(run.final_cost);
  }

  const auto [min_final, max_final] =
      std::minmax_element(final_costs.begin(), final_costs.end());

  const RunSummary& representative = runs.front();

  std::cout << "\n" << label << '\n';
  std::cout << "Initial cost                    : "
            << representative.initial_cost << '\n';
  std::cout << "Final cost                      : "
            << representative.final_cost << '\n';
  std::cout << "Iterations                      : "
            << representative.iterations << '\n';
  std::cout << "Converged                       : "
            << (representative.converged ? "yes" : "no") << '\n';
  std::cout << "Termination reason              : "
            << representative.termination_reason << '\n';
  std::cout << "Median linearization [ms]       : "
            << median(linearization) << '\n';
  std::cout << "Median linear solve [ms]        : "
            << median(solve) << '\n';
  std::cout << "Median optimizer wall [ms]      : "
            << median(wall) << '\n';
  std::cout << "Final-cost repetition spread    : "
            << (*max_final - *min_final) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc < 2 || argc > 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " path/to/dataset.g2o [max_iterations] [repetitions]\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];
    const std::size_t max_iterations =
        argc >= 3 ? static_cast<std::size_t>(std::stoul(argv[2])) : 30;
    const std::size_t repetitions =
        argc >= 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 3;

    if (max_iterations == 0) {
      throw std::invalid_argument("max_iterations must be greater than zero.");
    }
    if (repetitions == 0) {
      throw std::invalid_argument("repetitions must be greater than zero.");
    }

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    GaussNewtonOptions qr_options;
    qr_options.max_iterations = max_iterations;
    qr_options.linear_solver = GaussNewtonLinearSolver::kSparseQr;

    GaussNewtonOptions ldlt_options = qr_options;
    ldlt_options.linear_solver = GaussNewtonLinearSolver::kSymmetricLdlt;

    std::cout << std::setprecision(15);
    std::cout << "\nDataset                           : "
              << dataset_path << '\n';
    std::cout << "Poses                             : "
              << initialization.poses.size() << '\n';
    std::cout << "Edges                             : "
              << dataset.measurements.size() << '\n';
    std::cout << "Trajectory source                 : "
              << trajectorySourceName(initialization.source) << '\n';
    std::cout << "Max iterations                    : "
              << max_iterations << '\n';
    std::cout << "Timing repetitions                : "
              << repetitions << '\n';

    const std::vector<RunSummary> qr_runs =
        runSolver(
            initialization.poses,
            dataset.measurements,
            qr_options,
            repetitions);

    const std::vector<RunSummary> ldlt_runs =
        runSolver(
            initialization.poses,
            dataset.measurements,
            ldlt_options,
            repetitions);

    printSolverSummary("SparseQR", qr_runs);
    printSolverSummary("Symmetric LDLT", ldlt_runs);

    std::vector<double> qr_solve;
    std::vector<double> ldlt_solve;
    std::vector<double> qr_wall;
    std::vector<double> ldlt_wall;

    for (const RunSummary& run : qr_runs) {
      qr_solve.push_back(run.solve_ms);
      qr_wall.push_back(run.wall_ms);
    }
    for (const RunSummary& run : ldlt_runs) {
      ldlt_solve.push_back(run.solve_ms);
      ldlt_wall.push_back(run.wall_ms);
    }

    const double qr_final = qr_runs.front().final_cost;
    const double ldlt_final = ldlt_runs.front().final_cost;
    const double relative_final_cost_gap =
        std::abs(qr_final - ldlt_final) /
        std::max(1.0, std::max(std::abs(qr_final), std::abs(ldlt_final)));

    const double qr_solve_median = median(qr_solve);
    const double ldlt_solve_median = median(ldlt_solve);
    const double qr_wall_median = median(qr_wall);
    const double ldlt_wall_median = median(ldlt_wall);

    std::cout << "\nComparison\n";
    std::cout << "Relative final-cost gap          : "
              << relative_final_cost_gap << '\n';
    std::cout << "QR / LDLT solve-time ratio       : ";
    if (ldlt_solve_median > 0.0) {
      std::cout << qr_solve_median / ldlt_solve_median << '\n';
    } else {
      std::cout << "inf\n";
    }
    std::cout << "QR / LDLT wall-time ratio        : ";
    if (ldlt_wall_median > 0.0) {
      std::cout << qr_wall_median / ldlt_wall_median << '\n';
    } else {
      std::cout << "inf\n";
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_optimizer_solvers FAILED: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
