#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cora_symforce/gauss_newton.hpp"
#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/trajectory_io.hpp"

namespace {

namespace fs = std::filesystem;

double rotationAngle(const Eigen::Matrix2d& R) {
  return std::atan2(R(1, 0), R(0, 0));
}

void writeIterationSummary(
    const fs::path& path,
    const cora_symforce::GaussNewtonResult& result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Unable to open iteration summary: " + path.string());
  }

  out << std::setprecision(17);
  out << "iteration,cost,gradient_norm,step_norm,alpha,linearization_ms,"
         "solve_ms,cumulative_linearization_ms,cumulative_solve_ms\n";

  double cumulative_linearization = 0.0;
  double cumulative_solve = 0.0;

  out << 0 << ',' << result.initial_cost
      << ",0,0,0,0,0,0,0\n";

  for (const auto& iteration : result.iterations) {
    cumulative_linearization += iteration.linearization_time_ms;
    cumulative_solve += iteration.solve_time_ms;

    out << (iteration.iteration + 1) << ','
        << iteration.cost_after << ','
        << iteration.gradient_norm << ','
        << iteration.step_norm << ','
        << iteration.step_scale << ','
        << iteration.linearization_time_ms << ','
        << iteration.solve_time_ms << ','
        << cumulative_linearization << ','
        << cumulative_solve << '\n';
  }
}

void writeTrajectoryTrace(
    const fs::path& path,
    const cora_symforce::GaussNewtonResult& result) {
  if (result.trajectory_history.empty()) {
    throw std::runtime_error(
        "Trajectory history is empty. Enable record_trajectory_history.");
  }

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Unable to open trajectory trace: " + path.string());
  }

  out << std::setprecision(17);
  out << "iteration,pose_id,x,y,theta\n";

  for (std::size_t iteration = 0;
       iteration < result.trajectory_history.size();
       ++iteration) {
    const auto& poses = result.trajectory_history[iteration];
    for (std::size_t pose_id = 0; pose_id < poses.size(); ++pose_id) {
      out << iteration << ','
          << pose_id << ','
          << poses[pose_id].t.x() << ','
          << poses[pose_id].t.y() << ','
          << rotationAngle(poses[pose_id].R) << '\n';
    }
  }
}

void writeSolverOutputs(
    const fs::path& output_dir,
    const std::string& prefix,
    const cora_symforce::GaussNewtonResult& result) {
  writeIterationSummary(output_dir / (prefix + "_iterations.csv"), result);
  writeTrajectoryTrace(output_dir / (prefix + "_trajectory_trace.csv"), result);
  cora_symforce::writeTrajectoryCsv(
      (output_dir / (prefix + "_final.csv")).string(), result.poses);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc < 3 || argc > 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " path/to/dataset.g2o output_directory [max_iterations]\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];
    const fs::path output_dir = argv[2];
    const std::size_t max_iterations =
        argc == 4 ? static_cast<std::size_t>(std::stoul(argv[3])) : 30;

    if (max_iterations == 0) {
      throw std::invalid_argument("max_iterations must be greater than zero.");
    }

    fs::create_directories(output_dir);

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization = initializeTrajectory(dataset);

    GaussNewtonOptions qr_options;
    qr_options.max_iterations = max_iterations;
    qr_options.linear_solver = GaussNewtonLinearSolver::kSparseQr;
    qr_options.record_trajectory_history = true;

    GaussNewtonOptions ldlt_options = qr_options;
    ldlt_options.linear_solver = GaussNewtonLinearSolver::kSymmetricLdlt;

    const GaussNewtonResult qr = optimizeGaussNewton(
        initialization.poses, dataset.measurements, qr_options);
    const GaussNewtonResult ldlt = optimizeGaussNewton(
        initialization.poses, dataset.measurements, ldlt_options);

    writeSolverOutputs(output_dir, "qr", qr);
    writeSolverOutputs(output_dir, "ldlt", ldlt);

    std::ofstream meta(output_dir / "metadata.txt");
    if (!meta) {
      throw std::runtime_error("Unable to open metadata output.");
    }
    meta << std::setprecision(17);
    meta << "dataset=" << dataset_path << '\n';
    meta << "poses=" << initialization.poses.size() << '\n';
    meta << "edges=" << dataset.measurements.size() << '\n';
    meta << "trajectory_source=" << trajectorySourceName(initialization.source) << '\n';
    meta << "qr_initial_cost=" << qr.initial_cost << '\n';
    meta << "qr_final_cost=" << qr.final_cost << '\n';
    meta << "qr_iterations=" << qr.iterations.size() << '\n';
    meta << "qr_converged=" << (qr.converged ? 1 : 0) << '\n';
    meta << "qr_total_solve_ms=" << qr.total_solve_time_ms << '\n';
    meta << "ldlt_initial_cost=" << ldlt.initial_cost << '\n';
    meta << "ldlt_final_cost=" << ldlt.final_cost << '\n';
    meta << "ldlt_iterations=" << ldlt.iterations.size() << '\n';
    meta << "ldlt_converged=" << (ldlt.converged ? 1 : 0) << '\n';
    meta << "ldlt_total_solve_ms=" << ldlt.total_solve_time_ms << '\n';

    std::cout << std::setprecision(15);
    std::cout << "Dataset                : " << dataset_path << '\n';
    std::cout << "Output directory       : " << output_dir << '\n';
    std::cout << "QR final cost          : " << qr.final_cost << '\n';
    std::cout << "LDLT final cost        : " << ldlt.final_cost << '\n';
    std::cout << "QR iterations          : " << qr.iterations.size() << '\n';
    std::cout << "LDLT iterations        : " << ldlt.iterations.size() << '\n';
    std::cout << "QR solve total [ms]    : " << qr.total_solve_time_ms << '\n';
    std::cout << "LDLT solve total [ms]  : " << ldlt.total_solve_time_ms << '\n';
    std::cout << "Trace export complete.\n";

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "export_solver_validation FAILED: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
