#pragma once

#include <cstddef>

#include <Eigen/Core>

namespace cora_symforce {

struct RelativePoseMeasurement {
  std::size_t i;
  std::size_t j;

  Eigen::Matrix2d R;
  Eigen::Vector2d t;

  double kappa;
  double tau;
};

}  // namespace cora_symforce
