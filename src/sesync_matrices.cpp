#include "cora_symforce/sesync_matrices.hpp"

#include <cmath>
#include <stdexcept>

namespace cora_symforce {

namespace {

constexpr Eigen::Index kDim = 2;
constexpr Eigen::Index kDimSquared = kDim * kDim;

}  // namespace

SesyncMatrices buildSesyncMatrices(
    const std::vector<RelativePoseMeasurement>& measurements,
    std::size_t num_poses) {
  const Eigen::Index num_measurements =
      static_cast<Eigen::Index>(measurements.size());

  const Eigen::Index num_states =
      static_cast<Eigen::Index>(num_poses);

  SesyncMatrices matrices;

  matrices.B1 = Eigen::MatrixXd::Zero(
      kDim * num_measurements,
      kDim * num_states);

  matrices.B2 = Eigen::MatrixXd::Zero(
      kDim * num_measurements,
      kDimSquared * num_states);

  matrices.B3 = Eigen::MatrixXd::Zero(
      kDimSquared * num_measurements,
      kDimSquared * num_states);

  matrices.M = Eigen::MatrixXd::Zero(
      (kDim + 1) * num_states,
      (kDim + 1) * num_states);

  for (Eigen::Index e = 0; e < num_measurements; ++e) {
    const auto& measurement = measurements[e];

    if (measurement.i >= num_poses || measurement.j >= num_poses) {
      throw std::out_of_range(
          "measurement references an invalid pose index");
    }

    const Eigen::Index i =
        static_cast<Eigen::Index>(measurement.i);

    const Eigen::Index j =
        static_cast<Eigen::Index>(measurement.j);

    const double sqrt_tau = std::sqrt(measurement.tau);
    const double sqrt_kappa = std::sqrt(measurement.kappa);

    // B1: weighted translation difference t_j - t_i.
    for (Eigen::Index row = 0; row < kDim; ++row) {
      matrices.B1(
          e * kDim + row,
          i * kDim + row) = -sqrt_tau;

      matrices.B1(
          e * kDim + row,
          j * kDim + row) = sqrt_tau;
    }

    // B2: weighted -R_i * t_ij term.
    for (Eigen::Index column = 0; column < kDim; ++column) {
      for (Eigen::Index row = 0; row < kDim; ++row) {
        matrices.B2(
            e * kDim + row,
            i * kDimSquared + column * kDim + row) =
            -sqrt_tau * measurement.t(column);
      }
    }

    // B3: weighted R_j - R_i * R_ij term.
    for (Eigen::Index column = 0; column < kDim; ++column) {
      for (Eigen::Index row = 0; row < kDim; ++row) {
        const Eigen::Index output_index =
            e * kDimSquared + column * kDim + row;

        matrices.B3(
            output_index,
            j * kDimSquared + column * kDim + row) =
            sqrt_kappa;

        for (Eigen::Index inner = 0; inner < kDim; ++inner) {
          matrices.B3(
              output_index,
              i * kDimSquared + inner * kDim + row) +=
              -sqrt_kappa * measurement.R(inner, column);
        }
      }
    }

    // Translation Laplacian L(W^tau).
    matrices.M(i, i) += measurement.tau;
    matrices.M(j, j) += measurement.tau;
    matrices.M(i, j) -= measurement.tau;
    matrices.M(j, i) -= measurement.tau;

    // Translation-rotation coupling V and V^T.
    for (Eigen::Index k = 0; k < kDim; ++k) {
      const Eigen::Index rotation_index =
          num_states + i * kDim + k;

      const double value =
          measurement.tau * measurement.t(k);

      matrices.M(i, rotation_index) += value;
      matrices.M(j, rotation_index) -= value;

      matrices.M(rotation_index, i) += value;
      matrices.M(rotation_index, j) -= value;
    }

    // Rotational connection Laplacian L(G^rho).
    for (Eigen::Index k = 0; k < kDim; ++k) {
      matrices.M(
          num_states + i * kDim + k,
          num_states + i * kDim + k) +=
          measurement.kappa;

      matrices.M(
          num_states + j * kDim + k,
          num_states + j * kDim + k) +=
          measurement.kappa;
    }

    for (Eigen::Index row = 0; row < kDim; ++row) {
      for (Eigen::Index column = 0; column < kDim; ++column) {
        matrices.M(
            num_states + i * kDim + row,
            num_states + j * kDim + column) +=
            -measurement.kappa * measurement.R(row, column);

        matrices.M(
            num_states + j * kDim + row,
            num_states + i * kDim + column) +=
            -measurement.kappa * measurement.R(column, row);
      }
    }

    // Sigma block from the translational measurement.
    for (Eigen::Index row = 0; row < kDim; ++row) {
      for (Eigen::Index column = 0; column < kDim; ++column) {
        matrices.M(
            num_states + i * kDim + row,
            num_states + i * kDim + column) +=
            measurement.tau *
            measurement.t(row) *
            measurement.t(column);
      }
    }
  }

  return matrices;
}

Eigen::VectorXd stackTranslations(
    const std::vector<Pose2State>& poses) {
  Eigen::VectorXd stacked(kDim * poses.size());

  for (std::size_t i = 0; i < poses.size(); ++i) {
    stacked.segment<kDim>(kDim * i) = poses[i].t;
  }

  return stacked;
}

Eigen::VectorXd stackRotations(
    const std::vector<Pose2State>& poses) {
  Eigen::VectorXd stacked(kDimSquared * poses.size());

  for (std::size_t i = 0; i < poses.size(); ++i) {
    const Eigen::Map<const Eigen::Vector4d> vectorized_rotation(
        poses[i].R.data());

    stacked.segment<kDimSquared>(kDimSquared * i) =
        vectorized_rotation;
  }

  return stacked;
}

Eigen::MatrixXd buildExplicitStateMatrix(
    const std::vector<Pose2State>& poses) {
  const Eigen::Index num_states =
      static_cast<Eigen::Index>(poses.size());

  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(
      kDim,
      (kDim + 1) * num_states);

  for (Eigen::Index i = 0; i < num_states; ++i) {
    Y.col(i) = poses[i].t;

    Y.block(
        0,
        num_states + i * kDim,
        kDim,
        kDim) = poses[i].R;
  }

  return Y;
}

Eigen::VectorXd weightedTranslationResiduals(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements) {
  Eigen::VectorXd residuals(kDim * measurements.size());

  for (std::size_t e = 0; e < measurements.size(); ++e) {
    const auto& measurement = measurements[e];

    const Pose2State& pose_i = poses.at(measurement.i);
    const Pose2State& pose_j = poses.at(measurement.j);

    const Eigen::Vector2d error =
        pose_j.t -
        pose_i.t -
        pose_i.R * measurement.t;

    residuals.segment<kDim>(kDim * e) =
        std::sqrt(measurement.tau) * error;
  }

  return residuals;
}

Eigen::VectorXd weightedRotationResiduals(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements) {
  Eigen::VectorXd residuals(
      kDimSquared * measurements.size());

  for (std::size_t e = 0; e < measurements.size(); ++e) {
    const auto& measurement = measurements[e];

    const Pose2State& pose_i = poses.at(measurement.i);
    const Pose2State& pose_j = poses.at(measurement.j);

    const Eigen::Matrix2d error =
        pose_j.R -
        pose_i.R * measurement.R;

    const Eigen::Map<const Eigen::Vector4d> vectorized_error(
        error.data());

    residuals.segment<kDimSquared>(kDimSquared * e) =
        std::sqrt(measurement.kappa) * vectorized_error;
  }

  return residuals;
}

double evaluateExplicitQuadraticCost(
    const Eigen::MatrixXd& Y,
    const Eigen::MatrixXd& M) {
  return (Y * M * Y.transpose()).trace();
}

}  // namespace cora_symforce
