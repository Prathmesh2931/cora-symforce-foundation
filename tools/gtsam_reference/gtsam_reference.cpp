#include <gtsam/geometry/Pose2.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct IterationRow {
  std::size_t iteration{};
  double error_before{};
  double error_after{};
};

static void writeHistory(
    const std::string& path,
    const std::vector<IterationRow>& history) {

  std::ofstream out(path);

  out << "iteration,error_before,error_after,relative_drop\n";
  out << std::setprecision(17);

  for (const auto& row : history) {
    const double rel =
        std::abs(row.error_before - row.error_after) /
        std::max(1.0, std::abs(row.error_before));

    out
        << row.iteration << ","
        << row.error_before << ","
        << row.error_after << ","
        << rel << "\n";
  }
}

int main(int argc, char** argv) {

  if (argc < 3 || argc > 6) {
    std::cerr
        << "usage:\n"
        << "  gtsam_reference "
        << "<input.g2o> "
        << "<output_prefix> "
        << "[solver=lm] "
        << "[max_iterations=200] "
        << "[relative_tol=1e-9]\n";

    return 1;
  }

  const std::string input = argv[1];
  const std::string prefix = argv[2];

  const std::string solver =
      argc >= 4 ? argv[3] : "lm";

  const int max_iterations =
      argc >= 5 ? std::atoi(argv[4]) : 200;

  const double relative_tol =
      argc >= 6 ? std::atof(argv[5]) : 1e-9;

  // ---------------------------------------------------------
  // Load g2o graph
  // ---------------------------------------------------------

  const auto loaded =
      gtsam::readG2o(input, false);

  if (!loaded.first || !loaded.second) {
    std::cerr << "Failed to read g2o dataset.\n";
    return 2;
  }

  gtsam::NonlinearFactorGraph graph =
      *loaded.first;

  gtsam::Values initial =
      *loaded.second;

  if (initial.empty()) {
    std::cerr
        << "No VERTEX_SE2 initialization found.\n";
    return 3;
  }

  // ---------------------------------------------------------
  // Fix global SE(2) gauge
  // ---------------------------------------------------------

  const gtsam::Key anchor_key =
      initial.keys().front();

  const gtsam::Pose2 anchor_pose =
      initial.at<gtsam::Pose2>(anchor_key);

  graph.add(
      gtsam::PriorFactor<gtsam::Pose2>(
          anchor_key,
          anchor_pose,
          gtsam::noiseModel::Constrained::All(3)));

  const double initial_error =
      graph.error(initial);

  std::vector<IterationRow> history;

  gtsam::Values result;

  // ---------------------------------------------------------
  // Solve
  // ---------------------------------------------------------

  const auto t0 =
      std::chrono::steady_clock::now();

  if (solver == "gn") {

    gtsam::GaussNewtonParams params;

    params.setMaxIterations(max_iterations);
    params.setRelativeErrorTol(relative_tol);
    params.setAbsoluteErrorTol(0.0);
    params.setErrorTol(0.0);

    params.setVerbosity("SILENT");

    // Robust numerical reference for rank / conditioning issues.
    params.setLinearSolverType("MULTIFRONTAL_QR");

    params.iterationHook =
        [&](std::size_t iteration,
            double error_before,
            double error_after) {

          history.push_back({
              iteration,
              error_before,
              error_after
          });
        };

    gtsam::GaussNewtonOptimizer optimizer(
        graph,
        initial,
        params);

    result = optimizer.optimize();

  } else if (solver == "lm") {

    gtsam::LevenbergMarquardtParams params;

    params.setMaxIterations(max_iterations);
    params.setRelativeErrorTol(relative_tol);
    params.setAbsoluteErrorTol(0.0);
    params.setErrorTol(0.0);

    params.setVerbosity("SILENT");
    params.setVerbosityLM("SILENT");

    // Keep QR so our external reference is not relying on
    // an ill-conditioned normal-equation Cholesky solve.
    params.setLinearSolverType("MULTIFRONTAL_QR");

    params.iterationHook =
        [&](std::size_t iteration,
            double error_before,
            double error_after) {

          history.push_back({
              iteration,
              error_before,
              error_after
          });
        };

    gtsam::LevenbergMarquardtOptimizer optimizer(
        graph,
        initial,
        params);

    result = optimizer.optimize();

  } else {

    std::cerr
        << "Unknown solver: "
        << solver
        << "\nUse 'gn' or 'lm'.\n";

    return 4;
  }

  const auto t1 =
      std::chrono::steady_clock::now();

  const double wall_ms =
      std::chrono::duration<double, std::milli>(
          t1 - t0)
          .count();

  const double final_error =
      graph.error(result);

  // ---------------------------------------------------------
  // Export
  // ---------------------------------------------------------

  const std::string g2o_path =
      prefix + "_" + solver + "_optimized.g2o";

  const std::string csv_path =
      prefix + "_" + solver + "_iterations.csv";

  gtsam::writeG2o(
      graph,
      result,
      g2o_path);

  writeHistory(
      csv_path,
      history);

  // ---------------------------------------------------------
  // Summary
  // ---------------------------------------------------------

  std::cout << std::setprecision(15);

  std::cout
      << "Dataset             : "
      << input << "\n";

  std::cout
      << "Solver              : "
      << solver << "\n";

  std::cout
      << "Initial GTSAM error : "
      << initial_error << "\n";

  std::cout
      << "Final GTSAM error   : "
      << final_error << "\n";

  std::cout
      << "Iterations          : "
      << history.size() << "\n";

  std::cout
      << "Wall time [ms]      : "
      << wall_ms << "\n";

  std::cout
      << "Optimized g2o       : "
      << g2o_path << "\n";

  std::cout
      << "Iteration CSV       : "
      << csv_path << "\n";

  return 0;
}
