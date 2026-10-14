#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/toy_problem.hpp"
#include "cora_symforce/trajectory_io.hpp"

namespace {

bool checkNear(
    const double error,
    const double tolerance,
    const char* label) {
  std::cout << std::left << std::setw(46) << label
            << " error = " << std::scientific << error << '\n';

  if (error > tolerance) {
    std::cerr << "FAILED: " << label
              << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

double rotationError(
    const Eigen::Matrix2d& lhs,
    const Eigen::Matrix2d& rhs) {
  return (lhs - rhs).norm();
}

}  // namespace

int main() {
  using namespace cora_symforce;

  bool passed = true;

  // -------------------------------------------------------------------------
  // 1. Complete VERTEX_SE2 state must be preserved exactly.
  // -------------------------------------------------------------------------
  {
    const ToyProblem toy = makeToyProblem();

    G2oDataset dataset;
    dataset.poses = toy.initial_guess;
    dataset.measurements = toy.measurements;
    dataset.vertex_records = dataset.poses.size();
    dataset.has_complete_vertex_initialization = true;

    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    if (initialization.source != TrajectorySource::kDatasetVertices ||
        initialization.poses.size() != dataset.poses.size()) {
      std::cerr << "FAILED: complete dataset source/size mismatch.\n";
      passed = false;
    }

    double maximum_pose_error = 0.0;
    for (std::size_t i = 0; i < dataset.poses.size(); ++i) {
      maximum_pose_error = std::max(
          maximum_pose_error,
          rotationError(initialization.poses[i].R, dataset.poses[i].R));
      maximum_pose_error = std::max(
          maximum_pose_error,
          (initialization.poses[i].t - dataset.poses[i].t).norm());
    }

    passed &= checkNear(
        maximum_pose_error,
        1e-14,
        "Complete dataset trajectory preservation");
  }

  // -------------------------------------------------------------------------
  // 2. Edge-only graph must propagate both forward and reverse measurements.
  // -------------------------------------------------------------------------
  {
    std::vector<Pose2State> truth(4);
    truth[0] = {rotationMatrix(0.00), Eigen::Vector2d(0.0, 0.0)};
    truth[1] = {rotationMatrix(0.18), Eigen::Vector2d(1.1, 0.2)};
    truth[2] = {rotationMatrix(0.31), Eigen::Vector2d(2.0, 0.5)};
    truth[3] = {rotationMatrix(0.47), Eigen::Vector2d(2.7, 1.2)};

    G2oDataset dataset;
    dataset.poses.resize(truth.size());
    dataset.has_complete_vertex_initialization = false;

    // Root pose 0 reaches pose 1 through the reverse of edge 1 -> 0,
    // then reaches pose 2 forward, and pose 3 through the reverse of 3 -> 2.
    dataset.measurements = {
        makeRelativeMeasurement(truth, 1, 0, 4.0, 5.0),
        makeRelativeMeasurement(truth, 1, 2, 4.0, 5.0),
        makeRelativeMeasurement(truth, 3, 2, 4.0, 5.0),
    };

    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    if (initialization.source !=
            TrajectorySource::kPropagatedMeasurements ||
        initialization.connected_components != 1 ||
        initialization.tree_edges_used != 3) {
      std::cerr << "FAILED: edge-only trajectory metadata mismatch.\n";
      passed = false;
    }

    double maximum_pose_error = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
      maximum_pose_error = std::max(
          maximum_pose_error,
          rotationError(initialization.poses[i].R, truth[i].R));
      maximum_pose_error = std::max(
          maximum_pose_error,
          (initialization.poses[i].t - truth[i].t).norm());
    }

    passed &= checkNear(
        maximum_pose_error,
        1e-12,
        "Measurement-propagated trajectory");
  }

  // -------------------------------------------------------------------------
  // 3. Disconnected graphs are initialized as a deterministic spanning forest.
  // -------------------------------------------------------------------------
  {
    std::vector<Pose2State> truth(4);
    truth[0] = {rotationMatrix(0.00), Eigen::Vector2d(0.0, 0.0)};
    truth[1] = {rotationMatrix(0.10), Eigen::Vector2d(1.0, 0.0)};
    truth[2] = {rotationMatrix(0.00), Eigen::Vector2d(0.0, 0.0)};
    truth[3] = {rotationMatrix(-0.20), Eigen::Vector2d(0.5, 0.7)};

    G2oDataset dataset;
    dataset.poses.resize(4);
    dataset.measurements = {
        makeRelativeMeasurement(truth, 0, 1, 1.0, 1.0),
        makeRelativeMeasurement(truth, 2, 3, 1.0, 1.0),
    };

    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    if (initialization.connected_components != 2 ||
        initialization.tree_edges_used != 2) {
      std::cerr << "FAILED: disconnected graph component count mismatch.\n";
      passed = false;
    }
  }

  // -------------------------------------------------------------------------
  // 4. Export format sanity check.
  // -------------------------------------------------------------------------
  {
    const std::string csv_file = "test_trajectory.csv";
    const std::string g2o_file = "test_trajectory_vertices.g2o";

    const ToyProblem toy = makeToyProblem();
    writeTrajectoryCsv(csv_file, toy.initial_guess);
    writeTrajectoryVerticesG2o(g2o_file, toy.initial_guess);

    std::ifstream csv(csv_file);
    std::string header;
    std::getline(csv, header);
    if (header != "id,x,y,theta") {
      std::cerr << "FAILED: trajectory CSV header mismatch.\n";
      passed = false;
    }

    std::ifstream g2o(g2o_file);
    std::string first_token;
    g2o >> first_token;
    if (first_token != "VERTEX_SE2") {
      std::cerr << "FAILED: trajectory g2o export mismatch.\n";
      passed = false;
    }

    std::remove(csv_file.c_str());
    std::remove(g2o_file.c_str());
  }

  if (!passed) {
    std::cerr << "\nTrajectory initialization/export test FAILED.\n";
    return EXIT_FAILURE;
  }

  std::cout << "\nTrajectory initialization/export checks PASSED.\n";
  return EXIT_SUCCESS;
}
