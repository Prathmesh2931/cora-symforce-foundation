#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/gauss_newton.hpp"
#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/trajectory_io.hpp"

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc != 3 && argc != 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " path/to/dataset.g2o output_prefix [max_iterations]\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];
    const std::string output_prefix = argv[2];

    GaussNewtonOptions options;
    if (argc == 4) {
      options.max_iterations =
          static_cast<std::size_t>(std::stoul(argv[3]));
    }

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    const GaussNewtonResult result =
        optimizeGaussNewton(
            initialization.poses,
            dataset.measurements,
            options);

    const std::string csv_path =
        output_prefix + ".csv";
    const std::string vertices_path =
        output_prefix + "_vertices.g2o";

    writeTrajectoryCsv(csv_path, result.poses);
    writeTrajectoryVerticesG2o(vertices_path, result.poses);

    std::cout << std::setprecision(15);
    std::cout << "\nDataset                    : " << dataset_path << '\n';
    std::cout << "Poses                      : " << result.poses.size() << '\n';
    std::cout << "Edges                      : " << dataset.measurements.size() << '\n';
    std::cout << "Trajectory source          : "
              << trajectorySourceName(initialization.source) << '\n';
    std::cout << "Initial cost               : " << result.initial_cost << '\n';
    std::cout << "Final cost                 : " << result.final_cost << '\n';
    std::cout << "Cost reduction factor      : ";
    if (result.final_cost > 0.0) {
      std::cout << result.initial_cost / result.final_cost << '\n';
    } else {
      std::cout << "inf\n";
    }
    std::cout << "Iterations                 : " << result.iterations.size() << '\n';
    std::cout << "Converged                  : "
              << (result.converged ? "yes" : "no") << '\n';
    std::cout << "Termination reason         : "
              << result.termination_reason << '\n';
    std::cout << "Linearization total [ms]   : "
              << result.total_linearization_time_ms << '\n';
    std::cout << "SparseQR solve total [ms]  : "
              << result.total_solve_time_ms << '\n';
    std::cout << "CSV trajectory             : " << csv_path << '\n';
    std::cout << "VERTEX_SE2 trajectory      : " << vertices_path << '\n';

    std::cout
        << "\niter,cost_before,cost_after,gradient_norm,step_norm,alpha,rank,"
           "linearize_ms,solve_ms\n";

    for (const GaussNewtonIteration& iteration : result.iterations) {
      std::cout
          << iteration.iteration << ','
          << iteration.cost_before << ','
          << iteration.cost_after << ','
          << iteration.gradient_norm << ','
          << iteration.step_norm << ','
          << iteration.step_scale << ','
          << iteration.numerical_rank << ','
          << iteration.linearization_time_ms << ','
          << iteration.solve_time_ms << '\n';
    }

    // A non-converged run is still exported so it can be inspected. The app
    // returns failure to make scripts/CI notice that convergence was not met.
    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "optimize_g2o FAILED: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
