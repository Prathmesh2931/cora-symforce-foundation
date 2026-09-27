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

  // Oriented incidence matrix of the pose graph.
  // For edge i -> j, its column produces t_j - t_i.
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(n, m);

  // Translation measurement weights.
  // Using sqrt(tau) lets the weighted residual stay inside the norm.
  Eigen::MatrixXd sqrt_omega =
      Eigen::MatrixXd::Zero(m, m);

  // Translation measurement matrix.
  // R * T^T gives the -R_i * t_ij part of the residual.
  Eigen::MatrixXd T =
      Eigen::MatrixXd::Zero(m, kDim * n);

  // Rotational connection Laplacian used for the rotation part
  // of the PGO objective.
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

    // Edge orientation gives -t_i + t_j.
    A(i, e) = -1.0;
    A(j, e) = 1.0;

    sqrt_omega(e, e) =
        std::sqrt(measurement.tau);

    // Store -t_ij^T in the block belonging to R_i.
    // With R = [R_0 ... R_n], this gives -R_i * t_ij.
    for (Eigen::Index k = 0; k < kDim; ++k) {
      T(e, i * kDim + k) =
          -measurement.t(k);
    }

    // Diagonal blocks of L(G^rho).
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

    // Off-diagonal blocks:
    // L_ij = -kappa * R_ij
    // L_ji = -kappa * R_ij^T
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

  // Global translation is unobservable, so A has one redundant row.
  // Removing one row gives the reduced full-rank incidence system.
  const Eigen::MatrixXd A_red =
      A.topRows(n - 1);

  // B = A_red * Omega^(1/2)
  const Eigen::MatrixXd weighted_incidence =
      A_red * sqrt_omega;

  // B * B^T = A_red * Omega * A_red^T.
  // For a connected graph with positive weights this is invertible.
  const Eigen::MatrixXd normal_matrix =
      weighted_incidence *
      weighted_incidence.transpose();

  Eigen::LDLT<Eigen::MatrixXd> ldlt(normal_matrix);

  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor reduced incidence normal matrix");
  }

  // Projection that remains after eliminating the translation variables:
  //
  // Pi = I - B^T (B B^T)^-1 B
  //
  // LDLT is used to solve the linear system instead of forming
  // the inverse explicitly.
  const Eigen::MatrixXd Pi =
      Eigen::MatrixXd::Identity(m, m) -
      weighted_incidence.transpose() *
          ldlt.solve(weighted_incidence);

  // Reduced rotation-only matrix:
  //
  // Q = L(G^rho) + T^T Omega^(1/2) Pi Omega^(1/2) T
  //
  // L(G^rho) carries the rotation measurements.
  // The second term keeps the translation measurement information
  // after the translation states have been analytically eliminated.
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

  // Stack all rotation blocks horizontally:
  // R = [R_0 R_1 ... R_n]
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

  // For fixed rotations the translation residual is:
  //
  // B1 * t + B2 * r
  //
  // so B2 * r is constant while solving for t.
  const Eigen::VectorXd c =
      matrices.B2 * r;

  // Remove the t_0 block and use t_0 = 0 to fix the global
  // translation gauge.
  const Eigen::MatrixXd B1_reduced =
      matrices.B1.rightCols(
          kDim * (n - 1));

  // Solve the translation least-squares problem:
  //
  // min_t ||B1_reduced * t + c||^2
  const Eigen::VectorXd t_reduced =
      B1_reduced
          .colPivHouseholderQr()
          .solve(-c);

  Eigen::MatrixXd translations =
      Eigen::MatrixXd::Zero(kDim, n);

  // Column 0 remains zero because of the selected gauge.
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
  // Simplified SE-Sync objective after eliminating translations:
  // F(R) = trace(R Q R^T)
  return (R * Q * R.transpose()).trace();
}

}  // namespace cora_symforce
