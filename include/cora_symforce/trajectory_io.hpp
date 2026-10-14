#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

enum class TrajectorySource {
  kDatasetVertices,
  kPropagatedMeasurements,
};

struct TrajectoryInitialization {
  std::vector<Pose2State> poses;
  TrajectorySource source = TrajectorySource::kPropagatedMeasurements;
  std::size_t connected_components = 0;
  std::size_t tree_edges_used = 0;
};

// Return a readable label for logs and benchmark output.
const char* trajectorySourceName(TrajectorySource source);

// Construct an evaluation/initial trajectory for a loaded pose graph.
//
// If the dataset contains a complete VERTEX_SE2 initialization, those poses are
// returned unchanged. Otherwise a deterministic spanning-forest propagation is
// performed from the relative measurements. Each connected component is rooted
// at identity; tree propagation uses
//
//   R_j = R_i R_ij,
//   t_j = t_i + R_i t_ij.
//
// This is an initializer only. Loop closures are not optimized here.
TrajectoryInitialization initializeTrajectory(const G2oDataset& dataset);

// Export one row per pose:
//   id,x,y,theta
void writeTrajectoryCsv(
    const std::string& filename,
    const std::vector<Pose2State>& poses);

// Export only VERTEX_SE2 records. This is useful for visualization and for
// preserving an estimated trajectory separately from the original measurements.
void writeTrajectoryVerticesG2o(
    const std::string& filename,
    const std::vector<Pose2State>& poses);

}  // namespace cora_symforce
