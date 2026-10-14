#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/trajectory_io.hpp"

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " path/to/dataset.g2o output_prefix\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];
    const std::string output_prefix = argv[2];

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    const PgoEvaluation evaluation =
        evaluatePgo(initialization.poses, dataset.measurements);

    const std::string csv_path = output_prefix + ".csv";
    const std::string vertices_path =
        output_prefix + "_vertices.g2o";

    writeTrajectoryCsv(csv_path, initialization.poses);
    writeTrajectoryVerticesG2o(vertices_path, initialization.poses);

    std::cout << std::setprecision(15);
    std::cout << "\nDataset                    : " << dataset_path << '\n';
    std::cout << "Poses                      : " << initialization.poses.size() << '\n';
    std::cout << "Edges                      : " << dataset.measurements.size() << '\n';
    std::cout << "Trajectory source          : "
              << trajectorySourceName(initialization.source) << '\n';
    std::cout << "Connected components       : "
              << initialization.connected_components << '\n';
    std::cout << "Propagation tree edges     : "
              << initialization.tree_edges_used << '\n';
    std::cout << "Initial/evaluation PGO cost: " << evaluation.total << '\n';
    std::cout << "CSV trajectory             : " << csv_path << '\n';
    std::cout << "VERTEX_SE2 trajectory      : " << vertices_path << '\n';

    if (initialization.source ==
        TrajectorySource::kPropagatedMeasurements) {
      std::cout
          << "\nNOTE: This is a spanning-forest initializer, not an optimized "
             "trajectory.\n";
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "export_trajectory FAILED: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
