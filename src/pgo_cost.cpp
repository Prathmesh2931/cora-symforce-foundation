#include "cora_symforce/pgo_cost.hpp"

#include <stdexcept>

namespace cora_symforce {

EdgeCost evaluateEdge(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement) {
  if (measurement.i >= poses.size() || measurement.j >= poses.size()) {
    throw std::out_of_range("measurement references an invalid pose index");
  }

  const Pose2State& pose_i = poses[measurement.i];
  const Pose2State& pose_j = poses[measurement.j];
  
  const Eigen::Matrix2d rotation_error =
      pose_j.R - pose_i.R * measurement.R;

  const Eigen::Vector2d translation_error =
      pose_j.t - pose_i.t - pose_i.R * measurement.t;

  const double rotation_cost =
      measurement.kappa * rotation_error.squaredNorm();

  const double translation_cost =
      measurement.tau * translation_error.squaredNorm();

  return {
      measurement.i,
      measurement.j,
      rotation_cost,
      translation_cost,
  };
}

PgoEvaluation evaluatePgo(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements) {
  PgoEvaluation result;
  result.total = 0.0;
  result.edges.reserve(measurements.size());

  for (const RelativePoseMeasurement& measurement : measurements) {
    EdgeCost edge = evaluateEdge(poses, measurement);

    result.total += edge.total();
    result.edges.push_back(edge);
  }

  return result;
}

}  // namespace cora_symforce
