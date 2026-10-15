#include "cora_symforce/symmetric_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

namespace cora_symforce {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Clock = std::chrono::steady_clock;

struct ReducedJacobian {
  SparseMatrix matrix;
  std::vector<Eigen::Index> reduced_to_full;
};

double milliseconds(
    const Clock::time_point& start,
    const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("Cannot compute median of an empty sample.");
  }

  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;

  if (values.size() % 2 == 1) {
    return values[middle];
  }

  return 0.5 * (values[middle - 1] + values[middle]);
}

bool vectorIsFinite(const Eigen::VectorXd& vector) {
  return vector.array().isFinite().all();
}

SparseMatrix combineJacobians(const GraphLinearization& linearization) {
  const Eigen::Index rows = linearization.Jc.rows();
  const Eigen::Index constrained_cols = linearization.Jc.cols();
  const Eigen::Index free_cols = linearization.Jf.cols();

  if (linearization.Jf.rows() != rows) {
    throw std::invalid_argument("Jc and Jf must have the same row count.");
  }

  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(
      linearization.Jc.nonZeros() + linearization.Jf.nonZeros()));

  for (Eigen::Index outer = 0;
       outer < linearization.Jc.outerSize();
       ++outer) {
    for (SparseMatrix::InnerIterator it(linearization.Jc, outer); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  for (Eigen::Index outer = 0;
       outer < linearization.Jf.outerSize();
       ++outer) {
    for (SparseMatrix::InnerIterator it(linearization.Jf, outer); it; ++it) {
      triplets.emplace_back(
          it.row(), constrained_cols + it.col(), it.value());
    }
  }

  SparseMatrix J(rows, constrained_cols + free_cols);
  J.setFromTriplets(triplets.begin(), triplets.end());
  J.makeCompressed();
  return J;
}

ReducedJacobian removeTranslationGaugeColumns(
    const SparseMatrix& J,
    const Eigen::Index constrained_cols) {
  if (constrained_cols <= 0) {
    throw std::invalid_argument(
        "At least one constrained pose variable is required.");
  }

  const Eigen::Index fixed_tx = constrained_cols;
  const Eigen::Index fixed_ty = constrained_cols + 1;

  if (fixed_ty >= J.cols()) {
    throw std::invalid_argument(
        "Jacobian does not contain pose-0 translation columns.");
  }

  ReducedJacobian reduced;
  reduced.reduced_to_full.reserve(
      static_cast<std::size_t>(J.cols() - 2));

  std::vector<Eigen::Index> full_to_reduced(
      static_cast<std::size_t>(J.cols()), -1);

  Eigen::Index reduced_col = 0;
  for (Eigen::Index full_col = 0; full_col < J.cols(); ++full_col) {
    if (full_col == fixed_tx || full_col == fixed_ty) {
      continue;
    }

    full_to_reduced[static_cast<std::size_t>(full_col)] = reduced_col;
    reduced.reduced_to_full.push_back(full_col);
    ++reduced_col;
  }

  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(J.nonZeros()));

  for (Eigen::Index outer = 0; outer < J.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(J, outer); it; ++it) {
      const Eigen::Index mapped =
          full_to_reduced[static_cast<std::size_t>(it.col())];

      if (mapped >= 0) {
        triplets.emplace_back(it.row(), mapped, it.value());
      }
    }
  }

  reduced.matrix.resize(J.rows(), J.cols() - 2);
  reduced.matrix.setFromTriplets(triplets.begin(), triplets.end());
  reduced.matrix.makeCompressed();
  return reduced;
}

Eigen::VectorXd expandReducedDelta(
    const Eigen::VectorXd& reduced_delta,
    const ReducedJacobian& reduced,
    const Eigen::Index full_size) {
  if (reduced_delta.size() !=
      static_cast<Eigen::Index>(reduced.reduced_to_full.size())) {
    throw std::invalid_argument(
        "Reduced delta dimension does not match the gauge-reduced Jacobian.");
  }

  Eigen::VectorXd full = Eigen::VectorXd::Zero(full_size);

  for (Eigen::Index k = 0; k < reduced_delta.size(); ++k) {
    const Eigen::Index full_col =
        reduced.reduced_to_full[static_cast<std::size_t>(k)];
    full(full_col) = reduced_delta(k);
  }

  return full;
}

double sparseMaxAbs(const SparseMatrix& matrix) {
  double maximum = 0.0;

  for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      maximum = std::max(maximum, std::abs(it.value()));
    }
  }

  return maximum;
}

}  // namespace

LinearSolverComparison compareLinearSolvers(
    const GraphLinearization& linearization,
    const LinearSolverComparisonOptions& options) {
  if (options.repetitions == 0) {
    throw std::invalid_argument(
        "Linear solver comparison requires at least one repetition.");
  }

  if (linearization.residual.size() == 0 ||
      linearization.Jc.cols() == 0) {
    throw std::invalid_argument(
        "Linear solver comparison requires a non-empty graph linearization.");
  }

  const SparseMatrix J = combineJacobians(linearization);
  const Eigen::Index constrained_cols = linearization.Jc.cols();
  const Eigen::Index fixed_tx = constrained_cols;
  const Eigen::Index fixed_ty = constrained_cols + 1;
  const ReducedJacobian reduced =
      removeTranslationGaugeColumns(J, constrained_cols);

  LinearSolverComparison result;
  result.residual_rows = J.rows();
  result.state_columns = J.cols();
  result.reduced_state_columns = reduced.matrix.cols();
  result.jacobian_nonzeros = J.nonZeros();
  result.fixed_tx_column = fixed_tx;
  result.fixed_ty_column = fixed_ty;

  std::vector<double> qr_factor_times;
  std::vector<double> qr_solve_times;
  std::vector<double> normal_formation_times;
  std::vector<double> ldlt_factor_times;
  std::vector<double> ldlt_solve_times;

  qr_factor_times.reserve(options.repetitions);
  qr_solve_times.reserve(options.repetitions);
  normal_formation_times.reserve(options.repetitions);
  ldlt_factor_times.reserve(options.repetitions);
  ldlt_solve_times.reserve(options.repetitions);

  Eigen::VectorXd qr_delta;
  Eigen::VectorXd reduced_delta;
  SparseMatrix H;
  Eigen::VectorXd g;

  for (std::size_t repetition = 0;
       repetition < options.repetitions;
       ++repetition) {
    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;

    const auto qr_factor_start = Clock::now();
    qr.compute(J);
    const auto qr_factor_end = Clock::now();

    if (qr.info() != Eigen::Success) {
      throw std::runtime_error("SparseQR factorization failed.");
    }

    const auto qr_solve_start = Clock::now();
    const Eigen::VectorXd current_qr_delta =
        qr.solve(-linearization.residual);
    const auto qr_solve_end = Clock::now();

    if (qr.info() != Eigen::Success ||
        !vectorIsFinite(current_qr_delta)) {
      throw std::runtime_error(
          "SparseQR solve failed or produced a non-finite step.");
    }

    qr_factor_times.push_back(
        milliseconds(qr_factor_start, qr_factor_end));
    qr_solve_times.push_back(
        milliseconds(qr_solve_start, qr_solve_end));

    if (repetition == 0) {
      qr_delta = current_qr_delta;
      result.qr_rank = qr.rank();
    }

    const auto normal_start = Clock::now();
    SparseMatrix current_H =
        reduced.matrix.transpose() * reduced.matrix;
    current_H.makeCompressed();
    const Eigen::VectorXd current_g =
        reduced.matrix.transpose() * linearization.residual;
    const auto normal_end = Clock::now();

    Eigen::SimplicialLDLT<SparseMatrix> ldlt;
    const auto ldlt_factor_start = Clock::now();
    ldlt.compute(current_H);
    const auto ldlt_factor_end = Clock::now();

    if (ldlt.info() != Eigen::Success) {
      throw std::runtime_error(
          "SimplicialLDLT factorization failed after translation gauge removal.");
    }

    const auto ldlt_solve_start = Clock::now();
    const Eigen::VectorXd current_reduced_delta =
        ldlt.solve(-current_g);
    const auto ldlt_solve_end = Clock::now();

    if (ldlt.info() != Eigen::Success ||
        !vectorIsFinite(current_reduced_delta)) {
      throw std::runtime_error(
          "SimplicialLDLT solve failed or produced a non-finite step.");
    }

    normal_formation_times.push_back(
        milliseconds(normal_start, normal_end));
    ldlt_factor_times.push_back(
        milliseconds(ldlt_factor_start, ldlt_factor_end));
    ldlt_solve_times.push_back(
        milliseconds(ldlt_solve_start, ldlt_solve_end));

    if (repetition == 0) {
      H = std::move(current_H);
      g = current_g;
      reduced_delta = current_reduced_delta;
    }
  }

  result.qr_factorization_time_ms = median(qr_factor_times);
  result.qr_solve_time_ms = median(qr_solve_times);
  result.normal_formation_time_ms = median(normal_formation_times);
  result.ldlt_factorization_time_ms = median(ldlt_factor_times);
  result.ldlt_solve_time_ms = median(ldlt_solve_times);

  result.normal_nonzeros = H.nonZeros();

  const SparseMatrix H_transpose = H.transpose();
  const SparseMatrix H_difference = H - H_transpose;
  result.normal_symmetry_error = sparseMaxAbs(H_difference);

  const Eigen::VectorXd ldlt_delta =
      expandReducedDelta(reduced_delta, reduced, J.cols());

  const Eigen::VectorXd qr_prediction =
      J * qr_delta + linearization.residual;
  const Eigen::VectorXd ldlt_prediction =
      J * ldlt_delta + linearization.residual;

  result.qr_linearized_residual_norm = qr_prediction.norm();
  result.ldlt_linearized_residual_norm = ldlt_prediction.norm();
  result.linearized_residual_norm_gap = std::abs(
      result.qr_linearized_residual_norm -
      result.ldlt_linearized_residual_norm);
  result.prediction_difference_norm =
      (J * (qr_delta - ldlt_delta)).norm();

  const Eigen::VectorXd normal_residual = H * reduced_delta + g;
  result.ldlt_relative_normal_residual =
      normal_residual.norm() / std::max(1.0, g.norm());

  return result;
}

}  // namespace cora_symforce
