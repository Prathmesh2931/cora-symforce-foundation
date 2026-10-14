#include "cora_symforce/g2o_dataset.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace cora_symforce {
namespace {

void ensurePoseStorage(
    std::vector<Pose2State>& poses,
    std::vector<bool>& initialized,
    const std::size_t index) {
  if (index < poses.size()) {
    return;
  }

  const std::size_t new_size = index + 1;
  const std::size_t old_size = poses.size();
  poses.resize(new_size);
  initialized.resize(new_size, false);

  for (std::size_t k = old_size; k < new_size; ++k) {
    poses[k].R = Eigen::Matrix2d::Identity();
    poses[k].t = Eigen::Vector2d::Zero();
  }
}

}  // namespace

double seSyncTauFromTranslationInformation(
    const double I11,
    const double I12,
    const double I22) {
  const double determinant = I11 * I22 - I12 * I12;
  const double trace = I11 + I22;

  if (!(determinant > 0.0) || !(trace > 0.0)) {
    throw std::invalid_argument(
        "EDGE_SE2 translation information block must be positive definite.");
  }

  // For a 2x2 matrix A, trace(A^{-1}) = trace(A) / det(A).
  return 2.0 * determinant / trace;
}

G2oDataset loadG2oSe2(const std::string& filename) {
  std::ifstream input(filename);
  if (!input) {
    throw std::runtime_error("Unable to open g2o file: " + filename);
  }

  G2oDataset dataset;
  std::vector<bool> initialized;

  std::string line;
  std::size_t line_number = 0;

  while (std::getline(input, line)) {
    ++line_number;

    std::istringstream stream(line);
    std::string token;
    stream >> token;

    if (token.empty() || token[0] == '#') {
      continue;
    }

    if (token == "VERTEX_SE2") {
      std::size_t id = 0;
      double x = 0.0;
      double y = 0.0;
      double theta = 0.0;

      if (!(stream >> id >> x >> y >> theta)) {
        throw std::runtime_error(
            "Malformed VERTEX_SE2 at line " + std::to_string(line_number));
      }

      ensurePoseStorage(dataset.poses, initialized, id);
      dataset.poses[id].R = rotationMatrix(theta);
      dataset.poses[id].t << x, y;
      initialized[id] = true;
      ++dataset.vertex_records;
      continue;
    }

    if (token == "EDGE_SE2") {
      std::size_t i = 0;
      std::size_t j = 0;
      double dx = 0.0;
      double dy = 0.0;
      double dtheta = 0.0;
      double I11 = 0.0;
      double I12 = 0.0;
      double I13 = 0.0;
      double I22 = 0.0;
      double I23 = 0.0;
      double I33 = 0.0;

      if (!(stream >> i >> j >> dx >> dy >> dtheta
                   >> I11 >> I12 >> I13 >> I22 >> I23 >> I33)) {
        throw std::runtime_error(
            "Malformed EDGE_SE2 at line " + std::to_string(line_number));
      }

      (void)I13;
      (void)I23;

      ensurePoseStorage(dataset.poses, initialized, std::max(i, j));

      RelativePoseMeasurement measurement;
      measurement.i = i;
      measurement.j = j;
      measurement.R = rotationMatrix(dtheta);
      measurement.t << dx, dy;
      measurement.tau =
          seSyncTauFromTranslationInformation(I11, I12, I22);
      measurement.kappa = I33;

      dataset.measurements.push_back(measurement);
      continue;
    }

    if (token == "VERTEX_SE3:QUAT" || token == "EDGE_SE3:QUAT") {
      throw std::runtime_error(
          "This benchmark loader currently supports only 2D SE(2) g2o data; "
          "found " + token + " at line " + std::to_string(line_number));
    }

    // Common metadata lines do not affect this measurement-only benchmark.
    if (token == "FIX") {
      continue;
    }

    throw std::runtime_error(
        "Unrecognized g2o token '" + token + "' at line " +
        std::to_string(line_number));
  }

  if (dataset.measurements.empty()) {
    throw std::runtime_error("No EDGE_SE2 measurements found in: " + filename);
  }

  dataset.has_complete_vertex_initialization =
      !dataset.poses.empty() &&
      std::all_of(initialized.begin(), initialized.end(), [](const bool value) {
        return value;
      });

  return dataset;
}

}  // namespace cora_symforce
