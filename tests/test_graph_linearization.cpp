#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

bool checkNear(
    const double error,
    const double tolerance,
    const char* label) {
  std::cout << std::left << std::setw(44) << label
            << " error = " << std::scientific << error << '\n';

  if (error > tolerance) {
    std::cerr << "FAILED: " << label
              << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

double sparseSymmetryError(const Eigen::SparseMatrix<double>& matrix) {
  Eigen::MatrixXd dense(matrix);
  return (dense - dense.transpose()).cwiseAbs().maxCoeff();
}

cora_symforce::ToyProblem makeSyntheticProblem(const std::size_t n) {
  using namespace cora_symforce;

  ToyProblem problem;
  problem.ground_truth.reserve(n);
  problem.initial_guess.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const double s = static_cast<double>(i);
    const double theta = 0.012 * s + 0.03 * std::sin(0.11 * s);

    Pose2State truth;
    truth.R = rotationMatrix(theta);
    truth.t << 0.35 * s, 0.45 * std::sin(0.08 * s);
    problem.ground_truth.push_back(truth);

    Pose2State guess;
    guess.R = rotationMatrix(theta + 0.01 * std::sin(0.17 * s));
    guess.t = truth.t + Eigen::Vector2d(
        0.015 * std::cos(0.13 * s),
        -0.012 * std::sin(0.19 * s));
    problem.initial_guess.push_back(guess);
  }

  for (std::size_t i = 0; i + 1 < n; ++i) {
    problem.measurements.push_back(
        makeRelativeMeasurement(problem.ground_truth, i, i + 1, 8.0, 12.0));
  }

  // Deterministic loop closures every five poses.
  for (std::size_t i = 0; i + 10 < n; i += 5) {
    problem.measurements.push_back(
        makeRelativeMeasurement(problem.ground_truth, i, i + 10, 6.0, 9.0));
  }

  return problem;
}

bool validateGraph(
    const std::vector<cora_symforce::Pose2State>& poses,
    const std::vector<cora_symforce::RelativePoseMeasurement>& measurements,
    const char* label) {
  using namespace cora_symforce;

  const PgoEvaluation reference = evaluatePgo(poses, measurements);
  const PgoEvaluation generated = evaluateSymForcePgo(poses, measurements);
  const GraphLinearization linearization = linearizeGraph(poses, measurements);

  const Eigen::Index expected_rows =
      6 * static_cast<Eigen::Index>(measurements.size());
  const Eigen::Index expected_poses =
      static_cast<Eigen::Index>(poses.size());

  if (linearization.residual.size() != expected_rows ||
      linearization.Jc.rows() != expected_rows ||
      linearization.Jc.cols() != expected_poses ||
      linearization.Jf.rows() != expected_rows ||
      linearization.Jf.cols() != 2 * expected_poses) {
    std::cerr << "FAILED: " << label << " matrix dimensions are incorrect.\n";
    return false;
  }

  bool passed = true;
  passed &= checkNear(
      std::abs(reference.total - generated.total),
      1e-11,
      "Handwritten vs SymForce graph cost");
  passed &= checkNear(
      std::abs(linearization.cost() - generated.total),
      1e-11,
      "Residual stack squared norm vs cost");
  passed &= checkNear(
      sparseSymmetryError(linearization.Hcc),
      1e-12,
      "Hcc symmetry");
  passed &= checkNear(
      sparseSymmetryError(linearization.Hff),
      1e-12,
      "Hff symmetry");

  Eigen::VectorXd vx = Eigen::VectorXd::Zero(2 * expected_poses);
  Eigen::VectorXd vy = Eigen::VectorXd::Zero(2 * expected_poses);
  for (Eigen::Index i = 0; i < expected_poses; ++i) {
    vx(2 * i) = 1.0;
    vy(2 * i + 1) = 1.0;
  }

  passed &= checkNear(
      (linearization.Jf * vx).norm(),
      1e-12,
      "Global x-translation nullspace ||Jf vx||");
  passed &= checkNear(
      (linearization.Jf * vy).norm(),
      1e-12,
      "Global y-translation nullspace ||Jf vy||");

  // Verify local-to-global Jacobian scattering edge by edge.
  double max_scatter_error = 0.0;
  for (std::size_t edge_index = 0;
       edge_index < measurements.size();
       ++edge_index) {
    const RelativePoseMeasurement& measurement = measurements[edge_index];
    const SymForceEdgeJacobian local =
        evaluateSymForceEdgeJacobian(poses, measurement);

    const Eigen::Index row_offset =
        6 * static_cast<Eigen::Index>(edge_index);
    const Eigen::Index i = static_cast<Eigen::Index>(measurement.i);
    const Eigen::Index j = static_cast<Eigen::Index>(measurement.j);

    for (Eigen::Index row = 0; row < 6; ++row) {
      const Eigen::Index r = row_offset + row;
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jc.coeff(r, i) - local(row, 0)));
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jc.coeff(r, j) - local(row, 3)));
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jf.coeff(r, 2 * i) - local(row, 1)));
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jf.coeff(r, 2 * i + 1) - local(row, 2)));
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jf.coeff(r, 2 * j) - local(row, 4)));
      max_scatter_error = std::max(
          max_scatter_error,
          std::abs(linearization.Jf.coeff(r, 2 * j + 1) - local(row, 5)));
    }
  }

  passed &= checkNear(
      max_scatter_error,
      1e-12,
      "Local/global Jacobian scatter");

  std::cout << "\n[" << label << "] poses=" << poses.size()
            << " edges=" << measurements.size()
            << " cost=" << std::setprecision(15) << generated.total
            << " nnz(Jc)=" << linearization.Jc.nonZeros()
            << " nnz(Jf)=" << linearization.Jf.nonZeros()
            << "\n\n";

  return passed;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  bool passed = true;

  const ToyProblem toy = makeToyProblem();
  passed &= validateGraph(
      toy.initial_guess, toy.measurements, "three-pose toy graph");

  const ToyProblem synthetic = makeSyntheticProblem(100);
  passed &= validateGraph(
      synthetic.initial_guess,
      synthetic.measurements,
      "100-pose synthetic graph");

  if (!passed) {
    std::cerr << "\nGraph linearization test FAILED.\n";
    return EXIT_FAILURE;
  }

  std::cout << "Graph linearization checks PASSED.\n";
  return EXIT_SUCCESS;
}
