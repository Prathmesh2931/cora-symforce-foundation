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

  // B1 maps stacked translations to the weighted t_j - t_i terms.
  // Size: (d * m) x (d * n)
  matrices.B1 = Eigen::MatrixXd::Zero(
      kDim * num_measurements,
      kDim * num_states);

  // B2 maps vec(R) to the weighted -R_i * t_ij terms.
  // Together, B1 * t + B2 * r gives all translation residuals.
  matrices.B2 = Eigen::MatrixXd::Zero(
      kDim * num_measurements,
      kDimSquared * num_states);

  // B3 maps vec(R) to the weighted rotational residuals
  // vec(R_j - R_i * R_ij).
  matrices.B3 = Eigen::MatrixXd::Zero(
      kDimSquared * num_measurements,
      kDimSquared * num_states);

  // Explicit SE-Sync data matrix for
  // F(Y) = trace(Y * M * Y^T).
  //
  // Y = [t_0 ... t_n | R_0 ... R_n]
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

    // B1 edge block:
    //
    // sqrt(tau) * [-I  +I]
    //
    // so B1 * t gives sqrt(tau) * (t_j - t_i).
    for (Eigen::Index row = 0; row < kDim; ++row) {
      matrices.B1(
          e * kDim + row,
          i * kDim + row) = -sqrt_tau;

      matrices.B1(
          e * kDim + row,
          j * kDim + row) = sqrt_tau;
    }

    // B2 adds the rotation-dependent part of the translation residual:
    //
    // -sqrt(tau) * R_i * t_ij
    //
    // The indexing follows Eigen's column-major vec(R_i) layout.
    for (Eigen::Index column = 0; column < kDim; ++column) {
      for (Eigen::Index row = 0; row < kDim; ++row) {
        matrices.B2(
            e * kDim + row,
            i * kDimSquared + column * kDim + row) =
            -sqrt_tau * measurement.t(column);
      }
    }

    // B3 builds:
    //
    // sqrt(kappa) * vec(R_j - R_i * R_ij)
    //
    // The j block contributes +R_j, while the i block expands
    // the matrix product R_i * R_ij.
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
    //
    // This is the quadratic contribution of
    // tau * ||t_j - t_i||^2.
    matrices.M(i, i) += measurement.tau;
    matrices.M(j, j) += measurement.tau;
    matrices.M(i, j) -= measurement.tau;
    matrices.M(j, i) -= measurement.tau;

    // Translation-rotation coupling blocks V and V^T.
    //
    // These produce the cross term between (t_j - t_i)
    // and R_i * t_ij when the translation residual is expanded.
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

    // Diagonal blocks of the rotational connection Laplacian
    // L(G^rho).
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

    // Off-diagonal rotation blocks:
    //
    // L_ij = -kappa * R_ij
    // L_ji = -kappa * R_ij^T
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

    // Sigma block from the translation measurement:
    //
    // tau * t_ij * t_ij^T
    //
    // This gives the quadratic ||R_i * t_ij||^2 part of the
    // translation residual after expanding the squared norm.
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
  // t = [t_0; t_1; ...; t_n]
  Eigen::VectorXd stacked(kDim * poses.size());

  for (std::size_t i = 0; i < poses.size(); ++i) {
    stacked.segment<kDim>(kDim * i) = poses[i].t;
  }

  return stacked;
}

Eigen::VectorXd stackRotations(
    const std::vector<Pose2State>& poses) {
  // r = [vec(R_0); vec(R_1); ...; vec(R_n)]
  //
  // Eigen stores Matrix2d in column-major order by default, so
  // vec(R) follows [R00, R10, R01, R11]^T.
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

  // Explicit state used with M:
  //
  // Y = [t_0 ... t_n | R_0 ... R_n]
  //
  // Each translation uses one column, while each rotation uses
  // d consecutive columns.
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
  // Direct residual implementation used to validate
  // B1 * t + B2 * r.
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
  // Direct residual implementation used to validate B3 * r.
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
  // Same explicit PGO objective, written as a matrix quadratic form.
  return (Y * M * Y.transpose()).trace();
}

}  // namespace cora_symforce
