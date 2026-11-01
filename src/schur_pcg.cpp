#include "cora_symforce/schur_pcg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

namespace cora_symforce {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using TranslationLdlt = Eigen::SimplicialLDLT<SparseMatrix>;

struct PcgSolution {
  Eigen::VectorXd x;
  std::size_t iterations = 0;
  double relative_residual = 0.0;
};

SparseMatrix removeFirstTwoColumns(const SparseMatrix& matrix) {
  if (matrix.cols() < 2) {
    throw std::invalid_argument(
        "Cannot remove pose-0 translation columns from this matrix.");
  }

  SparseMatrix reduced(matrix.rows(), matrix.cols() - 2);
  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));

  for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      if (it.col() >= 2) {
        triplets.emplace_back(
            it.row(),
            it.col() - 2,
            it.value());
      }
    }
  }

  reduced.setFromTriplets(triplets.begin(), triplets.end());
  reduced.makeCompressed();
  return reduced;
}

SparseMatrix removeFirstTwoRowsAndColumns(
    const SparseMatrix& matrix) {
  if (matrix.rows() != matrix.cols() || matrix.rows() < 2) {
    throw std::invalid_argument(
        "Translation Hessian must be square and contain pose-0 coordinates.");
  }

  const Eigen::Index reduced_size = matrix.rows() - 2;
  SparseMatrix reduced(reduced_size, reduced_size);

  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));

  for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      if (it.row() >= 2 && it.col() >= 2) {
        triplets.emplace_back(
            it.row() - 2,
            it.col() - 2,
            it.value());
      }
    }
  }

  reduced.setFromTriplets(triplets.begin(), triplets.end());
  reduced.makeCompressed();
  return reduced;
}

void addDiagonalDamping(
    SparseMatrix& matrix,
    double damping) {
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    matrix.coeffRef(i, i) += damping;
  }
  matrix.makeCompressed();
}


// Construct the sparse block-Jacobi Schur surrogate:
//
//   S_tilde = Hcc - Hcf * D^{-1} * Hfc
//
// D contains the damped 2x2 diagonal translation blocks of Hff.
//
// Only the lower triangle is assembled because Eigen's
// IncompleteCholesky default UpLo is Lower.
//
// The actual matrix-free Schur operator remains unchanged.
SparseMatrix buildSparseSchurSurrogate(
    const SparseMatrix& Hcc,
    const SparseMatrix& Hcf,
    const SparseMatrix& Hff) {

  if (Hcc.rows() != Hcc.cols() ||
      Hcf.rows() != Hcc.rows() ||
      Hcf.cols() != Hff.rows() ||
      Hff.rows() != Hff.cols() ||
      Hff.rows() % 2 != 0) {
    throw std::invalid_argument(
        "Invalid dimensions for sparse Schur surrogate.");
  }

  const Eigen::Index nc = Hcc.rows();

  std::vector<Triplet> triplets;
  triplets.reserve(
      static_cast<std::size_t>(Hcc.nonZeros()));

  // Copy the lower triangle of the damped rotation Hessian.
  for (Eigen::Index col = 0; col < Hcc.outerSize(); ++col) {
    for (SparseMatrix::InnerIterator it(Hcc, col); it; ++it) {
      if (it.row() >= it.col()) {
        triplets.emplace_back(
            it.row(), it.col(), it.value());
      }
    }
  }

  struct RowCoupling {
    Eigen::Index row;
    Eigen::Vector2d value;
  };

  // Approximate translation elimination one 2x2 block at a time.
  for (Eigen::Index first_col = 0;
       first_col < Hff.cols();
       first_col += 2) {

    Eigen::Matrix2d D;

    D(0, 0) = Hff.coeff(first_col, first_col);
    D(0, 1) = Hff.coeff(first_col, first_col + 1);
    D(1, 0) = Hff.coeff(first_col + 1, first_col);
    D(1, 1) = Hff.coeff(first_col + 1, first_col + 1);

    D = (0.5 * (D + D.transpose())).eval();

    Eigen::LDLT<Eigen::Matrix2d> block_ldlt;
    block_ldlt.compute(D);

    if (block_ldlt.info() != Eigen::Success ||
        !block_ldlt.vectorD().array().isFinite().all() ||
        (block_ldlt.vectorD().array() <= 0.0).any()) {
      throw std::runtime_error(
          "P3 encountered a non-positive translation block.");
    }

    const Eigen::Matrix2d D_inverse =
        block_ldlt.solve(Eigen::Matrix2d::Identity());

    if (block_ldlt.info() != Eigen::Success ||
        !D_inverse.array().isFinite().all()) {
      throw std::runtime_error(
          "P3 translation block inverse solve failed.");
    }

    // Collect the nonzero rotational couplings to this translation block.
    // Both sparse columns are sorted by row index.
    std::vector<RowCoupling> entries;

    SparseMatrix::InnerIterator it0(Hcf, first_col);
    SparseMatrix::InnerIterator it1(Hcf, first_col + 1);

    while (it0 || it1) {
      Eigen::Index row = -1;
      double c0 = 0.0;
      double c1 = 0.0;

      if (it0 && (!it1 || it0.row() < it1.row())) {
        row = it0.row();
        c0 = it0.value();
        ++it0;

      } else if (it1 && (!it0 || it1.row() < it0.row())) {
        row = it1.row();
        c1 = it1.value();
        ++it1;

      } else {
        row = it0.row();
        c0 = it0.value();
        c1 = it1.value();
        ++it0;
        ++it1;
      }

      entries.push_back({
          row,
          Eigen::Vector2d(c0, c1)
      });
    }

    // Add the lower triangle of:
    //
    //   -C_k * D_k^{-1} * C_k^T
    //
    // This retains off-diagonal rotational couplings.
    for (std::size_t p = 0; p < entries.size(); ++p) {
      for (std::size_t q = 0; q <= p; ++q) {
        const Eigen::Vector2d weighted =
            D_inverse * entries[q].value;

        const double correction =
            entries[p].value.dot(weighted);

        if (!std::isfinite(correction)) {
          throw std::runtime_error(
              "P3 produced a non-finite Schur correction.");
        }

        if (correction != 0.0) {
          triplets.emplace_back(
              entries[p].row,
              entries[q].row,
              -correction);
        }
      }
    }
  }

  SparseMatrix surrogate(nc, nc);

  // Duplicate contributions are summed automatically.
  surrogate.setFromTriplets(
      triplets.begin(), triplets.end());

  surrogate.makeCompressed();

  for (Eigen::Index col = 0;
       col < surrogate.outerSize();
       ++col) {
    for (SparseMatrix::InnerIterator it(surrogate, col);
         it; ++it) {
      if (!std::isfinite(it.value())) {
        throw std::runtime_error(
            "P3 surrogate contains non-finite entries.");
      }
    }
  }

  return surrogate;
}

struct PreconditionerData {
  Eigen::VectorXd inverse_diagonal;
  double minimum_raw_diagonal = 0.0;
  std::size_t floored_entries = 0;
};

PreconditionerData buildPreconditionerDiagonal(
    const SparseMatrix& Hcc,
    const SparseMatrix& Hcf,
    const SparseMatrix& Hff,
    const SchurPcgOptions& options) {

  if (Hcc.rows() != Hcc.cols() ||
      Hcf.rows() != Hcc.rows() ||
      Hcf.cols() != Hff.rows() ||
      Hff.rows() != Hff.cols()) {
    throw std::invalid_argument(
        "Inconsistent matrix dimensions for Schur preconditioner.");
  }

  Eigen::VectorXd diagonal = Hcc.diagonal();

  if (diagonal.size() == 0 ||
      !diagonal.array().isFinite().all()) {
    throw std::runtime_error(
        "Preconditioner received an empty or non-finite diagonal.");
  }

  PreconditionerData result;

  if (options.preconditioner ==
      SchurPcgPreconditioner::kJacobiHcc) {

    // P0: original Jacobi preconditioner.
    if ((diagonal.array() <= 0.0).any()) {
      throw std::runtime_error(
          "P0 Jacobi preconditioner has a non-positive diagonal.");
    }

    result.minimum_raw_diagonal = diagonal.minCoeff();

  } else if (
      options.preconditioner ==
      SchurPcgPreconditioner::kSchurBlockJacobiDiagonal) {

    // P1: block-diagonal approximation of translation elimination.
    if (Hff.rows() % 2 != 0) {
      throw std::invalid_argument(
          "P1 requires two translation coordinates per pose.");
    }

    if (!(options.schur_diagonal_floor_relative > 0.0) ||
        !std::isfinite(options.schur_diagonal_floor_relative)) {
      throw std::invalid_argument(
          "P1 diagonal floor must be finite and positive.");
    }

    // Hcf ordering:
    // [t_1x, t_1y, t_2x, t_2y, ...].
    //
    // For each translation pose, extract its damped 2x2
    // diagonal block D_k and accumulate:
    //
    // diag(Hcf * D^{-1} * Hfc)_i
    //     = sum_k c_ik * D_k^{-1} * c_ik^T.
    //
    // Sparse column iterators avoid constructing a dense Schur matrix.

    for (Eigen::Index first_col = 0;
         first_col < Hff.cols();
         first_col += 2) {

      Eigen::Matrix2d D;

      D(0, 0) = Hff.coeff(first_col, first_col);
      D(0, 1) = Hff.coeff(first_col, first_col + 1);
      D(1, 0) = Hff.coeff(first_col + 1, first_col);
      D(1, 1) = Hff.coeff(first_col + 1, first_col + 1);

      // Remove roundoff-level asymmetry.
      D = (0.5 * (D + D.transpose())).eval();

      Eigen::LDLT<Eigen::Matrix2d> block_ldlt;
      block_ldlt.compute(D);

      if (block_ldlt.info() != Eigen::Success ||
          !block_ldlt.vectorD().array().isFinite().all() ||
          (block_ldlt.vectorD().array() <= 0.0).any()) {
        throw std::runtime_error(
            "P1 encountered a non-positive translation diagonal block.");
      }

      // Merge the sorted nonzero row indices of two sparse columns.
      SparseMatrix::InnerIterator it0(Hcf, first_col);
      SparseMatrix::InnerIterator it1(Hcf, first_col + 1);

      while (it0 || it1) {
        Eigen::Index row = -1;
        double c0 = 0.0;
        double c1 = 0.0;

        if (it0 && (!it1 || it0.row() < it1.row())) {
          row = it0.row();
          c0 = it0.value();
          ++it0;

        } else if (it1 && (!it0 || it1.row() < it0.row())) {
          row = it1.row();
          c1 = it1.value();
          ++it1;

        } else {
          row = it0.row();
          c0 = it0.value();
          c1 = it1.value();
          ++it0;
          ++it1;
        }

        const Eigen::Vector2d coupling(c0, c1);
        const Eigen::Vector2d solved =
            block_ldlt.solve(coupling);

        if (block_ldlt.info() != Eigen::Success ||
            !solved.array().isFinite().all()) {
          throw std::runtime_error(
              "P1 translation block solve failed.");
        }

        const double correction = coupling.dot(solved);

        if (!std::isfinite(correction)) {
          throw std::runtime_error(
              "P1 produced a non-finite Schur correction.");
        }

        diagonal(row) -= correction;
      }
    }

    if (!diagonal.array().isFinite().all()) {
      throw std::runtime_error(
          "P1 produced a non-finite approximate Schur diagonal.");
    }

    // Record the raw minimum BEFORE applying the positive floor.
    result.minimum_raw_diagonal = diagonal.minCoeff();

    const double scale = diagonal.size() > 0
        ? Hcc.diagonal().cwiseAbs().maxCoeff()
        : 0.0;

    if (!(scale > 0.0) || !std::isfinite(scale)) {
      throw std::runtime_error(
          "P1 could not determine a valid diagonal scale.");
    }

    const double floor =
        options.schur_diagonal_floor_relative * scale;

    if (!(floor > 0.0) || !std::isfinite(floor)) {
      throw std::runtime_error(
          "P1 computed an invalid positive diagonal floor.");
    }

    // A PCG diagonal preconditioner must be SPD.
    // Report every entry that required flooring.
    for (Eigen::Index i = 0; i < diagonal.size(); ++i) {
      if (diagonal(i) < floor) {
        diagonal(i) = floor;
        ++result.floored_entries;
      }
    }

  } else {
    throw std::invalid_argument(
        "Unknown Schur PCG preconditioner.");
  }

  if (!diagonal.array().isFinite().all() ||
      (diagonal.array() <= 0.0).any()) {
    throw std::runtime_error(
        "Schur PCG preconditioner diagonal is not positive.");
  }

  result.inverse_diagonal = diagonal.cwiseInverse();

  return result;
}

template <typename SchurApply, typename PreconditionerApply>
PcgSolution solvePcg(
    const SchurApply& apply,
    const PreconditionerApply& apply_preconditioner,
    const Eigen::VectorXd& rhs,
    const SchurPcgOptions& options) {
  if (rhs.size() == 0) {
    throw std::invalid_argument(
        "PCG received inconsistent or empty vector dimensions.");
  }

  const double rhs_norm = rhs.norm();
  const double target =
      options.absolute_tolerance +
      options.relative_tolerance * rhs_norm;

  PcgSolution result;
  result.x = Eigen::VectorXd::Zero(rhs.size());

  Eigen::VectorXd residual = rhs;
  double residual_norm = residual.norm();

  if (residual_norm <= target) {
    result.relative_residual =
        rhs_norm > 0.0 ? residual_norm / rhs_norm : 0.0;
    return result;
  }

  Eigen::VectorXd z =
      apply_preconditioner(residual);

  Eigen::VectorXd direction = z;
  double rho = residual.dot(z);

  if (!(rho > 0.0) || !std::isfinite(rho)) {
    throw std::runtime_error(
        "PCG encountered a non-positive initial preconditioned residual.");
  }

  for (std::size_t iteration = 0;
       iteration < options.max_iterations;
       ++iteration) {
    const Eigen::VectorXd applied = apply(direction);
    const double curvature = direction.dot(applied);

    if (!(curvature > 0.0) || !std::isfinite(curvature)) {
      throw std::runtime_error(
          "PCG encountered non-positive Schur curvature.");
    }

    const double alpha = rho / curvature;

    if (!std::isfinite(alpha)) {
      throw std::runtime_error(
          "PCG step length became non-finite.");
    }

    result.x.noalias() += alpha * direction;
    residual.noalias() -= alpha * applied;

    result.iterations = iteration + 1;
    residual_norm = residual.norm();

    if (!std::isfinite(residual_norm)) {
      throw std::runtime_error(
          "PCG residual became non-finite.");
    }

    if (residual_norm <= target) {
      // Recompute the true residual to detect recursive-residual drift.
      Eigen::VectorXd true_residual =
          rhs - apply(result.x);

      const double true_residual_norm =
          true_residual.norm();

      if (!std::isfinite(true_residual_norm)) {
        throw std::runtime_error(
            "PCG true residual became non-finite.");
      }

      if (true_residual_norm <= target) {
        result.relative_residual =
            rhs_norm > 0.0
                ? true_residual_norm / rhs_norm
                : 0.0;
        return result;
      }

      // The recursive residual was optimistic. Restart PCG from
      // the independently recomputed residual.
      residual = true_residual;
      residual_norm = true_residual_norm;

      z = apply_preconditioner(residual);
      rho = residual.dot(z);

      if (!(rho > 0.0) || !std::isfinite(rho)) {
        throw std::runtime_error(
            "PCG residual replacement produced a non-positive "
            "preconditioned residual.");
      }

      direction = z;
      continue;
    }

    z = apply_preconditioner(residual);
    const double next_rho = residual.dot(z);

    if (!(next_rho > 0.0) || !std::isfinite(next_rho)) {
      throw std::runtime_error(
          "PCG encountered a non-positive preconditioned residual.");
    }

    const double beta = next_rho / rho;

    direction = z + beta * direction;
    rho = next_rho;
  }

  const Eigen::VectorXd true_residual =
      rhs - apply(result.x);

  const double true_residual_norm =
      true_residual.norm();

  if (!std::isfinite(true_residual_norm)) {
    throw std::runtime_error(
        "PCG final true residual became non-finite.");
  }

  result.relative_residual =
      rhs_norm > 0.0 ? true_residual_norm / rhs_norm : 0.0;

  std::ostringstream message;
  message
      << "Schur PCG did not converge after "
      << result.iterations
      << " iterations; relative residual = "
      << result.relative_residual
      << ", requested relative tolerance = "
      << options.relative_tolerance
      << '.';

  throw std::runtime_error(message.str());
}

}  // namespace

SchurPcgResult solveSchurPcgReducedStep(
    const GraphLinearization& linearization,
    double damping,
    const SchurPcgOptions& options) {
  if (!(damping >= 0.0) || !std::isfinite(damping)) {
    throw std::invalid_argument(
        "Schur PCG damping must be finite and non-negative.");
  }

  if (options.max_iterations == 0) {
    throw std::invalid_argument(
        "Schur PCG max_iterations must be greater than zero.");
  }

  if (!(options.relative_tolerance >= 0.0) ||
      !std::isfinite(options.relative_tolerance) ||
      !(options.absolute_tolerance >= 0.0) ||
      !std::isfinite(options.absolute_tolerance) ||
      (options.relative_tolerance == 0.0 &&
       options.absolute_tolerance == 0.0)) {
    throw std::invalid_argument(
        "Schur PCG tolerances must be finite, non-negative, "
        "and not both zero.");
  }

  const Eigen::Index rotation_columns =
      linearization.Jc.cols();
  const Eigen::Index full_translation_columns =
      linearization.Jf.cols();

  if (rotation_columns <= 0 ||
      full_translation_columns < 2 ||
      linearization.Jc.rows() != linearization.residual.size() ||
      linearization.Jf.rows() != linearization.residual.size() ||
      linearization.Hcc.rows() != rotation_columns ||
      linearization.Hcc.cols() != rotation_columns ||
      linearization.Hcf.rows() != rotation_columns ||
      linearization.Hcf.cols() != full_translation_columns ||
      linearization.Hff.rows() != full_translation_columns ||
      linearization.Hff.cols() != full_translation_columns) {
    throw std::invalid_argument(
        "Graph linearization has inconsistent Schur block dimensions.");
  }

  const Eigen::Index translation_columns =
      full_translation_columns - 2;

  // Reuse the Hessian blocks assembled by linearizeGraph().
  // Only extract the gauge-reduced translation subblocks.
  SparseMatrix Hcc = linearization.Hcc;
  SparseMatrix Hcf =
      removeFirstTwoColumns(linearization.Hcf);
  SparseMatrix Hff =
      removeFirstTwoRowsAndColumns(linearization.Hff);

  Hcc.makeCompressed();
  Hcf.makeCompressed();
  Hff.makeCompressed();

  addDiagonalDamping(Hcc, damping);
  addDiagonalDamping(Hff, damping);

  const Eigen::VectorXd gc =
      linearization.Jc.transpose() * linearization.residual;

  const Eigen::VectorXd gf_full =
      linearization.Jf.transpose() * linearization.residual;

  const Eigen::VectorXd gf =
      gf_full.tail(translation_columns);

  // Factorize the damped translation block once. Every PCG Schur
  // matrix-vector product reuses this factorization.
  TranslationLdlt translation_ldlt;
  translation_ldlt.compute(Hff);

  if (translation_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "Schur PCG translation-block factorization failed.");
  }

  const Eigen::VectorXd y =
      translation_ldlt.solve(gf);

  if (translation_ldlt.info() != Eigen::Success ||
      !y.array().isFinite().all()) {
    throw std::runtime_error(
        "Schur PCG translation gradient solve failed.");
  }

  const Eigen::VectorXd rhs =
      -gc + Hcf * y;

  // Construct the selected preconditioner.
  // P0/P1 use diagonal scaling.
  // P3 uses incomplete Cholesky of a sparse Schur surrogate.
  PreconditionerData preconditioner;
  Eigen::IncompleteCholesky<double> p3_factor;

  const bool use_p3 =
      options.preconditioner ==
      SchurPcgPreconditioner::kSchurIncompleteCholesky;

  if (use_p3) {
    if (!(options.incomplete_cholesky_initial_shift > 0.0) ||
        !std::isfinite(options.incomplete_cholesky_initial_shift)) {
      throw std::invalid_argument(
          "P3 initial shift must be finite and positive.");
    }

    const SparseMatrix surrogate =
        buildSparseSchurSurrogate(Hcc, Hcf, Hff);

    const Eigen::VectorXd surrogate_diagonal =
        surrogate.diagonal();

    if (surrogate_diagonal.size() == 0 ||
        !surrogate_diagonal.array().isFinite().all()) {
      throw std::runtime_error(
          "P3 surrogate has an invalid diagonal.");
    }

    // Diagnostic only: no manual diagonal flooring is applied to P3.
    preconditioner.minimum_raw_diagonal =
        surrogate_diagonal.minCoeff();

    preconditioner.floored_entries = 0;

    p3_factor.setInitialShift(
        options.incomplete_cholesky_initial_shift);

    p3_factor.compute(surrogate);

    if (p3_factor.info() != Eigen::Success) {
      throw std::runtime_error(
          "P3 incomplete Cholesky factorization failed.");
    }

  } else {
    preconditioner =
        buildPreconditionerDiagonal(
            Hcc, Hcf, Hff, options);
  }

  // Matrix-free application:
  //
  // S_lambda * v =
  //   Hcc_lambda * v
  //   - Hcf * solve(Hff_lambda, Hcf.transpose() * v).
  //
  // No dense Schur complement or multi-RHS Hff solve is formed.
  const auto apply_schur =
      [&Hcc, &Hcf, &translation_ldlt](
          const Eigen::VectorXd& v) -> Eigen::VectorXd {
        if (v.size() != Hcc.cols()) {
          throw std::invalid_argument(
              "Schur operator input has an incorrect dimension.");
        }

        const Eigen::VectorXd hfc_v =
            Hcf.transpose() * v;

        const Eigen::VectorXd eliminated =
            translation_ldlt.solve(hfc_v);

        if (translation_ldlt.info() != Eigen::Success ||
            !eliminated.array().isFinite().all()) {
          throw std::runtime_error(
              "Schur operator translation solve failed.");
        }

        return Hcc * v - Hcf * eliminated;
      };

  const auto apply_preconditioner =
      [&preconditioner, &p3_factor, use_p3](
          const Eigen::VectorXd& residual) -> Eigen::VectorXd {

        Eigen::VectorXd result;

        if (use_p3) {
          result = p3_factor.solve(residual);
        } else {
          result =
              preconditioner.inverse_diagonal.cwiseProduct(residual);
        }

        if (result.size() != residual.size() ||
            !result.array().isFinite().all()) {
          throw std::runtime_error(
              "Schur PCG preconditioner application failed.");
        }

        return result;
      };

  const PcgSolution rotation_solution =
      solvePcg(
          apply_schur,
          apply_preconditioner,
          rhs,
          options);

  const Eigen::VectorXd delta_c =
      rotation_solution.x;

  // Translation back-substitution:
  //
  // (Hff + lambda I) * df = -(gf + Hfc * dc).
  const Eigen::VectorXd delta_f =
      translation_ldlt.solve(
          -gf - Hcf.transpose() * delta_c);

  if (translation_ldlt.info() != Eigen::Success ||
      !delta_f.array().isFinite().all()) {
    throw std::runtime_error(
        "Schur PCG translation back-substitution failed.");
  }

  SchurPcgResult result;
  result.reduced_delta.resize(
      rotation_columns + translation_columns);

  result.reduced_delta.head(rotation_columns) = delta_c;
  result.reduced_delta.tail(translation_columns) = delta_f;
  result.iterations = rotation_solution.iterations;
  result.relative_residual =
      rotation_solution.relative_residual;

  result.preconditioner_min_raw_diagonal =
      preconditioner.minimum_raw_diagonal;

  result.preconditioner_floored_diagonal_entries =
      preconditioner.floored_entries;

  return result;
}

}  // namespace cora_symforce
