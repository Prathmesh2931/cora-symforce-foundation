#include "cora_symforce/gauss_newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"

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

double rotationAngle(const Eigen::Matrix2d& R) {
  return std::atan2(R(1, 0), R(0, 0));
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
  const Eigen::Index fixed_tx = constrained_cols;
  const Eigen::Index fixed_ty = constrained_cols + 1;

  if (constrained_cols <= 0 || fixed_ty >= J.cols()) {
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
  Eigen::VectorXd full = Eigen::VectorXd::Zero(full_size);

  if (reduced_delta.size() !=
      static_cast<Eigen::Index>(reduced.reduced_to_full.size())) {
    throw std::invalid_argument(
        "Reduced delta dimension does not match reduced Jacobian.");
  }

  for (Eigen::Index k = 0; k < reduced_delta.size(); ++k) {
    full(reduced.reduced_to_full[static_cast<std::size_t>(k)]) =
        reduced_delta(k);
  }

  return full;
}

bool vectorIsFinite(const Eigen::VectorXd& vector) {
  return vector.array().isFinite().all();
}

std::vector<Pose2State> applyIncrement(
    const std::vector<Pose2State>& poses,
    const Eigen::VectorXd& delta,
    const double scale) {
  const Eigen::Index n = static_cast<Eigen::Index>(poses.size());

  if (delta.size() != 3 * n) {
    throw std::invalid_argument(
        "Gauss-Newton increment has an unexpected dimension.");
  }

  std::vector<Pose2State> updated = poses;

  for (Eigen::Index i = 0; i < n; ++i) {
    const std::size_t pose_index = static_cast<std::size_t>(i);
    const double theta = rotationAngle(poses[pose_index].R);

    updated[pose_index].R =
        rotationMatrix(theta + scale * delta(i));

    updated[pose_index].t.x() +=
        scale * delta(n + 2 * i);
    updated[pose_index].t.y() +=
        scale * delta(n + 2 * i + 1);
  }

  return updated;
}

}  // namespace

const char* gaussNewtonLinearSolverName(const GaussNewtonLinearSolver solver) {
  switch (solver) {
    case GaussNewtonLinearSolver::kSparseQr:
      return "sparse_qr";
    case GaussNewtonLinearSolver::kSymmetricLdlt:
      return "symmetric_ldlt";
  }
  return "unknown";
}

GaussNewtonResult optimizeGaussNewton(
    const std::vector<Pose2State>& initial_poses,
    const std::vector<RelativePoseMeasurement>& measurements,
    const GaussNewtonOptions& options) {
  if (initial_poses.empty()) {
    throw std::invalid_argument(
        "Gauss-Newton requires at least one pose.");
  }
  if (measurements.empty()) {
    throw std::invalid_argument(
        "Gauss-Newton requires at least one measurement.");
  }
  if (!(options.gradient_tolerance >= 0.0) ||
      !(options.step_tolerance >= 0.0) ||
      !(options.relative_cost_tolerance >= 0.0)) {
    throw std::invalid_argument(
        "Gauss-Newton tolerances must be non-negative.");
  }
  if (!(options.minimum_step_scale > 0.0) ||
      !(options.minimum_step_scale <= 1.0)) {
    throw std::invalid_argument(
        "minimum_step_scale must lie in (0, 1].");
  }

  GaussNewtonResult result;
  result.poses = initial_poses;
  if (options.record_trajectory_history) {
    result.trajectory_history.push_back(result.poses);
  }
  result.initial_cost =
      evaluateSymForcePgo(result.poses, measurements).total;
  result.final_cost = result.initial_cost;

  if (!std::isfinite(result.initial_cost)) {
    throw std::runtime_error("Initial PGO cost is not finite.");
  }

  for (std::size_t iteration = 0;
       iteration < options.max_iterations;
       ++iteration) {
    const auto linearization_start = Clock::now();

    const GraphLinearization linearization =
        linearizeGraph(result.poses, measurements);
    const SparseMatrix J = combineJacobians(linearization);
    const Eigen::VectorXd gradient =
        J.transpose() * linearization.residual;

    const auto linearization_end = Clock::now();
    const double linearization_ms =
        milliseconds(linearization_start, linearization_end);
    result.total_linearization_time_ms += linearization_ms;

    const double cost_before = linearization.cost();
    const double gradient_norm = gradient.norm();

    if (!std::isfinite(cost_before) || !std::isfinite(gradient_norm)) {
      throw std::runtime_error(
          "Non-finite cost or gradient during Gauss-Newton.");
    }

    if (gradient_norm <= options.gradient_tolerance) {
      result.converged = true;
      result.termination_reason = "gradient_tolerance";
      result.final_cost = cost_before;
      break;
    }

    const auto solve_start = Clock::now();

    Eigen::Index rank = -1;
    Eigen::VectorXd delta;

    if (options.linear_solver == GaussNewtonLinearSolver::kSparseQr) {
      Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
      qr.compute(J);

      if (qr.info() != Eigen::Success) {
        throw std::runtime_error("SparseQR factorization failed.");
      }

      rank = qr.rank();
      delta = qr.solve(-linearization.residual);

      if (qr.info() != Eigen::Success || !vectorIsFinite(delta)) {
        throw std::runtime_error(
            "SparseQR solve failed or produced a non-finite step.");
      }
    } else if (
        options.linear_solver == GaussNewtonLinearSolver::kSymmetricLdlt) {
      const ReducedJacobian reduced =
          removeTranslationGaugeColumns(J, linearization.Jc.cols());

      SparseMatrix H = reduced.matrix.transpose() * reduced.matrix;
      H.makeCompressed();
      const Eigen::VectorXd g =
          reduced.matrix.transpose() * linearization.residual;

      Eigen::SimplicialLDLT<SparseMatrix> ldlt;
      ldlt.compute(H);

      if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error(
            "SimplicialLDLT factorization failed during Gauss-Newton.");
      }

      const Eigen::VectorXd reduced_delta = ldlt.solve(-g);
      if (ldlt.info() != Eigen::Success ||
          !vectorIsFinite(reduced_delta)) {
        throw std::runtime_error(
            "SimplicialLDLT solve failed or produced a non-finite step.");
      }

      delta = expandReducedDelta(reduced_delta, reduced, J.cols());

      // SimplicialLDLT is not rank revealing.  After a successful solve this
      // records the dimension of the two-translation-gauge-reduced system.
      rank = reduced.matrix.cols();
    } else {
      throw std::invalid_argument("Unknown Gauss-Newton linear solver.");
    }

    const auto solve_end = Clock::now();
    const double solve_ms = milliseconds(solve_start, solve_end);
    result.total_solve_time_ms += solve_ms;

    const double full_step_norm = delta.norm();

    if (full_step_norm <= options.step_tolerance) {
      result.iterations.push_back({
          iteration,
          cost_before,
          cost_before,
          gradient_norm,
          full_step_norm,
          0.0,
          rank,
          linearization_ms,
          solve_ms,
      });

      result.converged = true;
      result.termination_reason = "step_tolerance";
      result.final_cost = cost_before;
      break;
    }

    double step_scale = 1.0;
    double accepted_cost = std::numeric_limits<double>::infinity();
    std::vector<Pose2State> accepted_poses;
    bool accepted = false;

    for (std::size_t backtrack = 0;
         backtrack <= options.max_backtracking_steps;
         ++backtrack) {
      if (step_scale < options.minimum_step_scale) {
        break;
      }

      std::vector<Pose2State> candidate =
          applyIncrement(result.poses, delta, step_scale);

      const double candidate_cost =
          evaluateSymForcePgo(candidate, measurements).total;

      const double acceptance_slack =
          1e-14 * std::max(1.0, std::abs(cost_before));

      if (std::isfinite(candidate_cost) &&
          candidate_cost <= cost_before + acceptance_slack) {
        accepted = true;
        accepted_cost = candidate_cost;
        accepted_poses = std::move(candidate);
        break;
      }

      step_scale *= 0.5;
    }

    if (!accepted) {
      result.converged = false;
      result.termination_reason = "line_search_failed";
      result.final_cost = cost_before;
      break;
    }

    const double accepted_step_norm =
        step_scale * full_step_norm;

    const double relative_cost_decrease =
        (cost_before - accepted_cost) /
        std::max(1.0, std::abs(cost_before));

    result.iterations.push_back({
        iteration,
        cost_before,
        accepted_cost,
        gradient_norm,
        accepted_step_norm,
        step_scale,
        rank,
        linearization_ms,
        solve_ms,
    });

    result.poses = std::move(accepted_poses);
    result.final_cost = accepted_cost;

    if (options.record_trajectory_history) {
      result.trajectory_history.push_back(result.poses);
    }

    if (accepted_step_norm <= options.step_tolerance) {
      result.converged = true;
      result.termination_reason = "step_tolerance";
      break;
    }

    if (relative_cost_decrease <= options.relative_cost_tolerance) {
      result.converged = true;
      result.termination_reason = "relative_cost_tolerance";
      break;
    }
  }

  if (result.termination_reason.empty()) {
    result.final_cost =
        evaluateSymForcePgo(result.poses, measurements).total;
    result.converged = false;
    result.termination_reason = "max_iterations";
  }

  return result;
}

}  // namespace cora_symforce
