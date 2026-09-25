#include "cora_symforce/simplified_pgo.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/QR>

namespace cora_symforce {

namespace {

constexpr Eigen::Index kDim = 2;

}  // namespace

SimplifiedProblem buildSimplifiedProblem(
    const std::vector<RelativePoseMeasurement>& measurements,
    std::size_t num_poses) {
  if (num_poses < 2) {
    throw std::invalid_argument(
        "simplified PGO requires at least two poses");
  }

  const Eigen::Index n =
      static_cast<Eigen::Index>(num_poses);

  const Eigen::Index m =
      static_cast<Eigen::Index>(measurements.size());

  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(n, m);

  Eigen::MatrixXd sqrt_omega =
      Eigen::MatrixXd::Zero(m, m);

  Eigen::MatrixXd T =
      Eigen::MatrixXd::Zero(m, kDim * n);

  Eigen::MatrixXd L_grho =
      Eigen::MatrixXd::Zero(kDim * n, kDim * n);

  for (Eigen::Index e = 0; e < m; ++e) {
    const auto& measurement = measurements[e];

    if (measurement.i >= num_poses ||
        measurement.j >= num_poses) {
      throw std::out_of_range(
          "measurement references an invalid pose index");
    }

    const Eigen::Index i =
        static_cast<Eigen::Index>(measurement.i);

    const Eigen::Index j =
        static_cast<Eigen::Index>(measurement.j);

    A(i, e) = -1.0;
    A(j, e) = 1.0;

    sqrt_omega(e, e) =
        std::sqrt(measurement.tau);

    for (Eigen::Index k = 0; k < kDim; ++k) {
      T(e, i * kDim + k) =
          -measurement.t(k);
    }

    for (Eigen::Index k = 0; k < kDim; ++k) {
      L_grho(
          i * kDim + k,
          i * kDim + k) +=
          measurement.kappa;

      L_grho(
          j * kDim + k,
          j * kDim + k) +=
          measurement.kappa;
    }

    for (Eigen::Index row = 0; row < kDim; ++row) {
      for (Eigen::Index column = 0;
           column < kDim;
           ++column) {
        L_grho(
            i * kDim + row,
            j * kDim + column) +=
            -measurement.kappa *
            measurement.R(row, column);

        L_grho(
            j * kDim + row,
            i * kDim + column) +=
            -measurement.kappa *
            measurement.R(column, row);
      }
    }
  }

  const Eigen::MatrixXd A_red =
      A.topRows(n - 1);

  const Eigen::MatrixXd weighted_incidence =
      A_red * sqrt_omega;

  const Eigen::MatrixXd normal_matrix =
      weighted_incidence *
      weighted_incidence.transpose();

  Eigen::LDLT<Eigen::MatrixXd> ldlt(normal_matrix);

  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor reduced incidence normal matrix");
  }

  const Eigen::MatrixXd Pi =
      Eigen::MatrixXd::Identity(m, m) -
      weighted_incidence.transpose() *
          ldlt.solve(weighted_incidence);

  const Eigen::MatrixXd Q =
      L_grho +
      T.transpose() *
          sqrt_omega *
          Pi *
          sqrt_omega *
          T;

  return {
      Pi,
      Q,
  };
}

Eigen::MatrixXd buildRotationStateMatrix(
    const std::vector<Pose2State>& poses) {
  const Eigen::Index n =
      static_cast<Eigen::Index>(poses.size());

  Eigen::MatrixXd R(kDim, kDim * n);

  for (Eigen::Index i = 0; i < n; ++i) {
    R.block(
        0,
        i * kDim,
        kDim,
        kDim) = poses[i].R;
  }

  return R;
}

Eigen::MatrixXd recoverOptimalTranslations(
    const std::vector<Pose2State>& poses,
    const SesyncMatrices& matrices) {
  const Eigen::Index n =
      static_cast<Eigen::Index>(poses.size());

  if (n < 2) {
    throw std::invalid_argument(
        "translation recovery requires at least two poses");
  }

  const Eigen::VectorXd r =
      stackRotations(poses);

  const Eigen::VectorXd c =
      matrices.B2 * r;

  const Eigen::MatrixXd B1_reduced =
      matrices.B1.rightCols(
          kDim * (n - 1));

  const Eigen::VectorXd t_reduced =
      B1_reduced
          .colPivHouseholderQr()
          .solve(-c);

  Eigen::MatrixXd translations =
      Eigen::MatrixXd::Zero(kDim, n);

  for (Eigen::Index i = 1; i < n; ++i) {
    translations.col(i) =
        t_reduced.segment<kDim>(
            kDim * (i - 1));
  }

  return translations;
}

double evaluateSimplifiedQuadraticCost(
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& Q) {
  return (R * Q * R.transpose()).trace();
}

}  // namespace cora_symforce
