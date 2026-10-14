#include "cora_symforce/trajectory_io.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cora_symforce {
namespace {

struct AdjacencyEntry {
  std::size_t measurement_index = 0;
  std::size_t neighbor = 0;
  bool forward = true;
};

double rotationAngle(const Eigen::Matrix2d& R) {
  return std::atan2(R(1, 0), R(0, 0));
}

std::vector<std::vector<AdjacencyEntry>> buildAdjacency(
    const G2oDataset& dataset) {
  std::vector<std::vector<AdjacencyEntry>> adjacency(dataset.poses.size());

  for (std::size_t k = 0; k < dataset.measurements.size(); ++k) {
    const RelativePoseMeasurement& measurement = dataset.measurements[k];

    if (measurement.i >= dataset.poses.size() ||
        measurement.j >= dataset.poses.size()) {
      throw std::out_of_range(
          "RelativePoseMeasurement references an invalid pose index.");
    }

    adjacency[measurement.i].push_back({k, measurement.j, true});
    adjacency[measurement.j].push_back({k, measurement.i, false});
  }

  return adjacency;
}

std::size_t countConnectedComponents(
    const std::vector<std::vector<AdjacencyEntry>>& adjacency) {
  std::vector<bool> visited(adjacency.size(), false);
  std::size_t components = 0;

  for (std::size_t root = 0; root < adjacency.size(); ++root) {
    if (visited[root]) {
      continue;
    }

    ++components;
    std::queue<std::size_t> queue;
    queue.push(root);
    visited[root] = true;

    while (!queue.empty()) {
      const std::size_t current = queue.front();
      queue.pop();

      for (const AdjacencyEntry& entry : adjacency[current]) {
        if (!visited[entry.neighbor]) {
          visited[entry.neighbor] = true;
          queue.push(entry.neighbor);
        }
      }
    }
  }

  return components;
}

}  // namespace

const char* trajectorySourceName(const TrajectorySource source) {
  switch (source) {
    case TrajectorySource::kDatasetVertices:
      return "dataset_vertices";
    case TrajectorySource::kPropagatedMeasurements:
      return "propagated_measurements";
  }

  return "unknown";
}

TrajectoryInitialization initializeTrajectory(const G2oDataset& dataset) {
  if (dataset.poses.empty()) {
    throw std::invalid_argument("Cannot initialize an empty pose graph.");
  }

  const auto adjacency = buildAdjacency(dataset);

  TrajectoryInitialization result;
  result.connected_components = countConnectedComponents(adjacency);

  if (dataset.has_complete_vertex_initialization) {
    result.poses = dataset.poses;
    result.source = TrajectorySource::kDatasetVertices;
    return result;
  }

  result.poses.resize(dataset.poses.size());
  std::vector<bool> initialized(dataset.poses.size(), false);
  result.source = TrajectorySource::kPropagatedMeasurements;

  for (std::size_t root = 0; root < result.poses.size(); ++root) {
    if (initialized[root]) {
      continue;
    }

    result.poses[root].R = Eigen::Matrix2d::Identity();
    result.poses[root].t = Eigen::Vector2d::Zero();
    initialized[root] = true;

    std::queue<std::size_t> queue;
    queue.push(root);

    while (!queue.empty()) {
      const std::size_t current = queue.front();
      queue.pop();

      for (const AdjacencyEntry& entry : adjacency[current]) {
        if (initialized[entry.neighbor]) {
          continue;
        }

        const RelativePoseMeasurement& measurement =
            dataset.measurements[entry.measurement_index];

        const Pose2State& current_pose = result.poses[current];
        Pose2State neighbor_pose;

        if (entry.forward) {
          // current == i, neighbor == j.
          neighbor_pose.R = current_pose.R * measurement.R;
          neighbor_pose.t =
              current_pose.t + current_pose.R * measurement.t;
        } else {
          // current == j, neighbor == i.
          // R_j = R_i R_ij  =>  R_i = R_j R_ij^T.
          neighbor_pose.R = current_pose.R * measurement.R.transpose();
          neighbor_pose.t =
              current_pose.t - neighbor_pose.R * measurement.t;
        }

        result.poses[entry.neighbor] = neighbor_pose;
        initialized[entry.neighbor] = true;
        ++result.tree_edges_used;
        queue.push(entry.neighbor);
      }
    }
  }

  return result;
}

void writeTrajectoryCsv(
    const std::string& filename,
    const std::vector<Pose2State>& poses) {
  std::ofstream output(filename);
  if (!output) {
    throw std::runtime_error("Unable to open trajectory CSV: " + filename);
  }

  output << std::setprecision(17);
  output << "id,x,y,theta\n";

  for (std::size_t i = 0; i < poses.size(); ++i) {
    output << i << ','
           << poses[i].t.x() << ','
           << poses[i].t.y() << ','
           << rotationAngle(poses[i].R) << '\n';
  }
}

void writeTrajectoryVerticesG2o(
    const std::string& filename,
    const std::vector<Pose2State>& poses) {
  std::ofstream output(filename);
  if (!output) {
    throw std::runtime_error(
        "Unable to open trajectory VERTEX_SE2 file: " + filename);
  }

  output << std::setprecision(17);

  for (std::size_t i = 0; i < poses.size(); ++i) {
    output << "VERTEX_SE2 " << i << ' '
           << poses[i].t.x() << ' '
           << poses[i].t.y() << ' '
           << rotationAngle(poses[i].R) << '\n';
  }
}

}  // namespace cora_symforce
