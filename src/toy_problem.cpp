#include "cora_symforce/toy_problem.hpp"

namespace cora_symforce {

RelativePoseMeasurement makeRelativeMeasurement(
    const std::vector<Pose2State>& poses,
    std::size_t i,
    std::size_t j,
    double kappa,
    double tau) {
  const Pose2State& pose_i = poses.at(i);
  const Pose2State& pose_j = poses.at(j);

  const Eigen::Matrix2d R_ij = pose_i.R.transpose() * pose_j.R;

  const Eigen::Vector2d t_ij =
      pose_i.R.transpose() * (pose_j.t - pose_i.t);

  return {
      i,
      j,
      R_ij,
      t_ij,
      kappa,
      tau,
  };
}

ToyProblem makeToyProblem() {
  ToyProblem problem;

  problem.ground_truth = {
      {
          rotationMatrix(0.00),
          Eigen::Vector2d(0.0, 0.0),
      },
      {
          rotationMatrix(0.25),
          Eigen::Vector2d(1.0, 0.2),
      },
      {
          rotationMatrix(0.50),
          Eigen::Vector2d(2.0, 0.1),
      },
  };

  problem.measurements = {
      makeRelativeMeasurement(
          problem.ground_truth, 0, 1, 10.0, 20.0),
      makeRelativeMeasurement(
          problem.ground_truth, 1, 2, 12.0, 18.0),
      makeRelativeMeasurement(
          problem.ground_truth, 0, 2, 8.0, 15.0),
  };

  problem.initial_guess = {
      problem.ground_truth[0],
      {
          rotationMatrix(0.30),
          Eigen::Vector2d(1.04, 0.17),
      },
      {
          rotationMatrix(0.47),
          Eigen::Vector2d(1.96, 0.14),
      },
  };

  return problem;
}

}  // namespace cora_symforce
