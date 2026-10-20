#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/dataset.h>

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr
        << "usage: gtsam_cost_eval "
        << "<measurement_graph.g2o> "
        << "<trajectory.g2o>\n";
    return 1;
  }

  const auto graph_loaded =
      gtsam::readG2o(argv[1], false);

  const auto trajectory_loaded =
      gtsam::readG2o(argv[2], false);

  if (!graph_loaded.first ||
      !trajectory_loaded.second) {
    std::cerr << "Failed to load input.\n";
    return 2;
  }

  const auto& graph =
      *graph_loaded.first;

  const auto& values =
      *trajectory_loaded.second;

  const double error =
      graph.error(values);

  std::cout << std::setprecision(15);

  std::cout
      << "Factors           : "
      << graph.size() << "\n";

  std::cout
      << "Trajectory poses  : "
      << values.size() << "\n";

  std::cout
      << "GTSAM graph error : "
      << error << "\n";

  std::cout
      << "2 * GTSAM error   : "
      << 2.0 * error << "\n";

  return 0;
}
