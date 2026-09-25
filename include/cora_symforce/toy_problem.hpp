#pragma once

#include <vector>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pose2.hpp"

namespace cora_symforce {

struct ToyProblem {
  std::vector<Pose2State> ground_truth;
  std::vector<RelativePoseMeasurement> measurements;
  std::vector<Pose2State> initial_guess;
};

RelativePoseMeasurement makeRelativeMeasurement(
    const std::vector<Pose2State>& poses,
    std::size_t i,
    std::size_t j,
    double kappa,
    double tau);

ToyProblem makeToyProblem();

}  // namespace cora_symforce
