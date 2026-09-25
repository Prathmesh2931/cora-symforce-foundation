#pragma once

#include <cstddef>
#include <vector>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct EdgeCost {
  std::size_t i;
  std::size_t j;

  double rotation;
  double translation;

  double total() const {
    return rotation + translation;
  }
};

struct PgoEvaluation {
  double total;
  std::vector<EdgeCost> edges;
};

EdgeCost evaluateEdge(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement);

PgoEvaluation evaluatePgo(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements);

}  // namespace cora_symforce
