#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "edge_jacobian.h"
#include "edge_residual.h"

namespace {

using Matrix2 = Eigen::Matrix2d;
using Vector2 = Eigen::Vector2d;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

Matrix2 rotationMatrix(const double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  Matrix2 R;
  R << c, -s,
       s,  c;
  return R;
}

Vector6 residualFromState(
    const Vector6& state,
    const Matrix2& measured_R_ij,
    const Vector2& measured_t_ij,
    const double kappa,
    const double tau) {
  Vector2 t_i;
  t_i << state(1), state(2);

  Vector2 t_j;
  t_j << state(4), state(5);

  return cora_symforce_generated::EdgeResidual<double>(
      state(0),
      t_i,
      state(3),
      t_j,
      measured_R_ij,
      measured_t_ij,
      kappa,
      tau);
}

double costFromState(
    const Vector6& state,
    const Matrix2& measured_R_ij,
    const Vector2& measured_t_ij,
    const double kappa,
    const double tau) {
  return residualFromState(
             state, measured_R_ij, measured_t_ij, kappa, tau)
      .squaredNorm();
}

bool checkNear(
    const double error,
    const double tolerance,
    const char* label) {
  std::cout << std::left << std::setw(42) << label
            << " error = " << std::scientific << error << '\n';

  if (error > tolerance) {
    std::cerr << "FAILED: " << label
              << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  constexpr double epsilon = 1e-6;

  Vector6 state;
  state << 0.23, 0.31, -0.18, 0.47, 1.27, 0.22;

  const Matrix2 measured_R_ij = rotationMatrix(0.21);
  Vector2 measured_t_ij;
  measured_t_ij << 0.93, 0.16;

  const double kappa = 4.7;
  const double tau = 3.2;

  Vector2 t_i;
  t_i << state(1), state(2);
  Vector2 t_j;
  t_j << state(4), state(5);

  const Vector6 residual = residualFromState(
      state, measured_R_ij, measured_t_ij, kappa, tau);

  const Matrix6 J = cora_symforce_generated::EdgeJacobian<double>(
      state(0),
      t_i,
      state(3),
      t_j,
      measured_R_ij,
      measured_t_ij,
      kappa,
      tau);

  const Matrix6 H = J.transpose() * J;
  const Vector6 g = J.transpose() * residual;

  const double h_symmetry_error =
      (H - H.transpose()).cwiseAbs().maxCoeff();

  // F = r^T r, therefore grad(F) = 2 J^T r.
  Vector6 fd_gradient;
  for (Eigen::Index k = 0; k < state.size(); ++k) {
    Vector6 plus = state;
    Vector6 minus = state;
    plus(k) += epsilon;
    minus(k) -= epsilon;

    fd_gradient(k) =
        (costFromState(plus, measured_R_ij, measured_t_ij, kappa, tau) -
         costFromState(minus, measured_R_ij, measured_t_ij, kappa, tau)) /
        (2.0 * epsilon);
  }

  const double gradient_error =
      (2.0 * g - fd_gradient).cwiseAbs().maxCoeff();

  // Independent directional derivative check for the residual itself.
  Vector6 direction;
  direction << 0.3, -0.4, 0.2, -0.1, 0.5, -0.6;
  direction.normalize();

  const Vector6 directional_fd =
      (residualFromState(
           state + epsilon * direction,
           measured_R_ij,
           measured_t_ij,
           kappa,
           tau) -
       residualFromState(
           state - epsilon * direction,
           measured_R_ij,
           measured_t_ij,
           kappa,
           tau)) /
      (2.0 * epsilon);

  const double directional_error =
      (J * direction - directional_fd).cwiseAbs().maxCoeff();

  std::cout << std::setprecision(15);
  std::cout << "\nCost             : " << residual.squaredNorm() << '\n';
  std::cout << "||J^T r||        : " << g.norm() << "\n\n";

  bool passed = true;
  passed &= checkNear(h_symmetry_error, 1e-12, "J^T J symmetry");
  passed &= checkNear(gradient_error, 1e-7, "2 J^T r vs FD cost gradient");
  passed &= checkNear(directional_error, 1e-7, "J v vs FD residual derivative");

  if (!passed) {
    std::cerr << "\nSymForce linearization test FAILED.\n";
    return EXIT_FAILURE;
  }

  std::cout << "\nSymForce linearization checks PASSED.\n";
  return EXIT_SUCCESS;
}
