#pragma once

#include <cmath>

#include <Eigen/Core>

namespace cora_symforce {

struct Pose2State {
  Eigen::Matrix2d R;
  Eigen::Vector2d t;
};

inline Eigen::Matrix2d rotationMatrix(double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  Eigen::Matrix2d R;
  R << c, -s,
       s,  c;

  return R;
}

}  // namespace cora_symforce
