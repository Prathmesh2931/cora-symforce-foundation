#include "cora_symforce/gauss_newton.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"

namespace cora_symforce {
namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Clock = std::chrono::steady_clock;

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

    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
    qr.compute(J);

    if (qr.info() != Eigen::Success) {
      throw std::runtime_error("SparseQR factorization failed.");
    }

    const Eigen::Index rank = qr.rank();
    const Eigen::VectorXd delta =
        qr.solve(-linearization.residual);

    const auto solve_end = Clock::now();
    const double solve_ms = milliseconds(solve_start, solve_end);
    result.total_solve_time_ms += solve_ms;

    if (qr.info() != Eigen::Success || !vectorIsFinite(delta)) {
      throw std::runtime_error(
          "SparseQR solve failed or produced a non-finite step.");
    }

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
