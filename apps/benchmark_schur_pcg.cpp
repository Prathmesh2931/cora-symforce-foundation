#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/schur_pcg.hpp"
#include "cora_symforce/trajectory_io.hpp"

namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Clock = std::chrono::steady_clock;

using cora_symforce::GraphLinearization;
using cora_symforce::SchurPcgOptions;
using cora_symforce::SchurPcgPreconditioner;
using cora_symforce::SchurPcgResult;

struct TimedSolve {
  SchurPcgResult result;
  double solve_time_ms = 0.0;
  double normal_relative_residual = 0.0;
};

SparseMatrix removeFirstTwoColumns(const SparseMatrix& matrix) {
  if (matrix.cols() < 2) {
    throw std::invalid_argument(
        "Cannot remove gauge translation columns.");
  }

  SparseMatrix reduced(matrix.rows(), matrix.cols() - 2);
  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<std::size_t>(matrix.nonZeros()));

  for (Eigen::Index outer = 0; outer < matrix.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(matrix, outer); it; ++it) {
      if (it.col() >= 2) {
        triplets.emplace_back(
            it.row(), it.col() - 2, it.value());
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
        "Translation Hessian must be square and contain gauge coordinates.");
  }

  SparseMatrix reduced(
      matrix.rows() - 2, matrix.cols() - 2);

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

void addDiagonalDamping(SparseMatrix& matrix, double damping) {
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    matrix.coeffRef(i, i) += damping;
  }
  matrix.makeCompressed();
}

// Evaluate the full gauge-reduced normal-equation residual:
//
// [Hcc+lambdaI  Hcf          ] [dc] + [gc] = 0
// [Hcf^T        Hff+lambdaI  ] [df]   [gf]
//
// This is deliberately outside the timed solver region.
double fullNormalRelativeResidual(
    const GraphLinearization& linearization,
    double damping,
    const Eigen::VectorXd& delta) {

  const Eigen::Index nc = linearization.Jc.cols();
  const Eigen::Index nf = linearization.Jf.cols() - 2;

  if (delta.size() != nc + nf) {
    throw std::invalid_argument(
        "PCG solution has an unexpected reduced dimension.");
  }

  SparseMatrix Hcc = linearization.Hcc;
  SparseMatrix Hcf = removeFirstTwoColumns(linearization.Hcf);
  SparseMatrix Hff =
      removeFirstTwoRowsAndColumns(linearization.Hff);

  addDiagonalDamping(Hcc, damping);
  addDiagonalDamping(Hff, damping);

  const Eigen::VectorXd gc =
      linearization.Jc.transpose() * linearization.residual;

  const Eigen::VectorXd gf_full =
      linearization.Jf.transpose() * linearization.residual;

  const Eigen::VectorXd gf = gf_full.tail(nf);

  const Eigen::VectorXd dc = delta.head(nc);
  const Eigen::VectorXd df = delta.tail(nf);

  const Eigen::VectorXd rc =
      Hcc * dc + Hcf * df + gc;

  const Eigen::VectorXd rf =
      Hcf.transpose() * dc + Hff * df + gf;

  const double numerator =
      std::sqrt(rc.squaredNorm() + rf.squaredNorm());

  const double denominator = std::max(
      1.0,
      std::sqrt(gc.squaredNorm() + gf.squaredNorm()));

  return numerator / denominator;
}

TimedSolve runOne(
    const GraphLinearization& linearization,
    double damping,
    SchurPcgOptions options) {

  TimedSolve sample;

  const auto start = Clock::now();

  sample.result =
      cora_symforce::solveSchurPcgReducedStep(
          linearization, damping, options);

  const auto finish = Clock::now();

  sample.solve_time_ms =
      std::chrono::duration<double, std::milli>(
          finish - start).count();

  sample.normal_relative_residual =
      fullNormalRelativeResidual(
          linearization,
          damping,
          sample.result.reduced_delta);

  return sample;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("Cannot compute median of empty samples.");
  }

  std::sort(values.begin(), values.end());

  const std::size_t middle = values.size() / 2;

  if (values.size() % 2 == 0) {
    return 0.5 * (values[middle - 1] + values[middle]);
  }

  return values[middle];
}

void printSummary(
    const char* name,
    const std::vector<TimedSolve>& samples) {

  std::vector<double> times;
  std::vector<double> iterations;

  double max_pcg_residual = 0.0;
  double max_normal_residual = 0.0;
  double minimum_raw_diagonal =
      std::numeric_limits<double>::infinity();

  std::size_t maximum_floored_entries = 0;
  std::size_t minimum_iterations =
      std::numeric_limits<std::size_t>::max();
  std::size_t maximum_iterations = 0;

  for (const TimedSolve& sample : samples) {
    times.push_back(sample.solve_time_ms);
    iterations.push_back(
        static_cast<double>(sample.result.iterations));

    max_pcg_residual = std::max(
        max_pcg_residual,
        sample.result.relative_residual);

    max_normal_residual = std::max(
        max_normal_residual,
        sample.normal_relative_residual);

    minimum_raw_diagonal = std::min(
        minimum_raw_diagonal,
        sample.result.preconditioner_min_raw_diagonal);

    maximum_floored_entries = std::max(
        maximum_floored_entries,
        sample.result.preconditioner_floored_diagonal_entries);

    minimum_iterations = std::min(
        minimum_iterations, sample.result.iterations);

    maximum_iterations = std::max(
        maximum_iterations, sample.result.iterations);
  }

  std::cout
      << name << '\n'
      << "  Median complete solve [ms] : " << median(times) << '\n'
      << "  Median PCG iterations      : " << median(iterations) << '\n'
      << "  Iteration range            : "
      << minimum_iterations << " - " << maximum_iterations << '\n'
      << "  Max true PCG residual      : " << max_pcg_residual << '\n'
      << "  Max full normal residual   : " << max_normal_residual << '\n'
      << "  Minimum raw diagonal       : " << minimum_raw_diagonal << '\n'
      << "  Maximum floored entries    : "
      << maximum_floored_entries << '\n';
}


bool runDampingExperiment(
    const GraphLinearization& linearization,
    double damping,
    std::size_t repetitions,
    std::size_t warmup_runs) {

  const std::vector<const char*> names = {
      "P0: Jacobi Hcc",
      "P1: Schur block-Jacobi",
      "P3: Schur incomplete Cholesky"
  };

  std::vector<SchurPcgOptions> configurations(3);

  configurations[0].max_iterations = 5000;
  configurations[0].relative_tolerance = 1e-10;
  configurations[0].absolute_tolerance = 1e-12;
  configurations[0].preconditioner =
      SchurPcgPreconditioner::kJacobiHcc;

  configurations[1] = configurations[0];
  configurations[1].preconditioner =
      SchurPcgPreconditioner::kSchurBlockJacobiDiagonal;

  configurations[2] = configurations[0];
  configurations[2].preconditioner =
      SchurPcgPreconditioner::kSchurIncompleteCholesky;
  configurations[2].incomplete_cholesky_initial_shift = 1e-3;

  // Warm up each implementation before measuring.
  for (std::size_t k = 0; k < configurations.size(); ++k) {
    for (std::size_t i = 0; i < warmup_runs; ++i) {
      (void)runOne(
          linearization,
          damping,
          configurations[k]);
    }
  }

  std::vector<std::vector<TimedSolve>> samples(3);

  for (auto& group : samples) {
    group.reserve(repetitions);
  }

  // Rotate execution order between repetitions.
  // Each repetition contains exactly one solve per preconditioner.
  for (std::size_t repetition = 0;
       repetition < repetitions;
       ++repetition) {

    for (std::size_t offset = 0;
         offset < configurations.size();
         ++offset) {

      const std::size_t k =
          (repetition + offset) % configurations.size();

      samples[k].push_back(
          runOne(
              linearization,
              damping,
              configurations[k]));
    }
  }

  std::cout << "\n----------------------------------------\n";
  std::cout << "Damping: " << damping << '\n';
  std::cout << "----------------------------------------\n";

  for (std::size_t k = 0; k < configurations.size(); ++k) {
    printSummary(names[k], samples[k]);
    std::cout << '\n';
  }

  // Compare P1 and P3 against P0 for each corresponding repetition.
  double maximum_p0_p1_gap = 0.0;
  double maximum_p0_p3_gap = 0.0;

  for (std::size_t repetition = 0;
       repetition < repetitions;
       ++repetition) {

    const Eigen::VectorXd& reference =
        samples[0][repetition].result.reduced_delta;

    const double p1_gap =
        (reference - samples[1][repetition].result.reduced_delta).norm() /
        std::max(1.0, reference.norm());

    const double p3_gap =
        (reference - samples[2][repetition].result.reduced_delta).norm() /
        std::max(1.0, reference.norm());

    maximum_p0_p1_gap = std::max(maximum_p0_p1_gap, p1_gap);
    maximum_p0_p3_gap = std::max(maximum_p0_p3_gap, p3_gap);
  }

  std::cout
      << "Maximum P0/P1 solution gap : "
      << maximum_p0_p1_gap << '\n'
      << "Maximum P0/P3 solution gap : "
      << maximum_p0_p3_gap << '\n';

  bool passed = true;
  double maximum_normal_residual = 0.0;
  double maximum_pcg_residual = 0.0;

  for (const auto& group : samples) {
    for (const TimedSolve& sample : group) {

      if (!sample.result.reduced_delta.array().isFinite().all() ||
          !std::isfinite(sample.result.relative_residual) ||
          !std::isfinite(sample.normal_relative_residual) ||
          !std::isfinite(sample.solve_time_ms)) {
        passed = false;
      }

      maximum_normal_residual = std::max(
          maximum_normal_residual,
          sample.normal_relative_residual);

      maximum_pcg_residual = std::max(
          maximum_pcg_residual,
          sample.result.relative_residual);
    }
  }

  std::cout
      << "Maximum true PCG residual       : "
      << maximum_pcg_residual << '\n'
      << "Maximum full normal residual    : "
      << maximum_normal_residual << '\n';

  if (maximum_p0_p1_gap > 1e-6 ||
      maximum_p0_p3_gap > 1e-6) {
    std::cerr
        << "FAILED: preconditioner solutions disagree excessively.\n";
    passed = false;
  }

  if (maximum_normal_residual > 1e-7) {
    std::cerr
        << "FAILED: full normal-equation residual is too large.\n";
    passed = false;
  }

  if (maximum_pcg_residual > 1e-8) {
    std::cerr
        << "FAILED: true PCG residual is too large.\n";
    passed = false;
  }

  if (!passed) {
    std::cerr << "Damping experiment FAILED.\n";
  } else {
    std::cout << "Damping experiment: PASSED\n";
  }

  return passed;
}


}  // namespace

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc < 2 || argc > 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " dataset.g2o [repetitions=5] [warmup_runs=1]\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];

    const std::size_t repetitions =
        argc >= 3 ? std::stoul(argv[2]) : 5;

    const std::size_t warmup_runs =
        argc >= 4 ? std::stoul(argv[3]) : 1;

    if (repetitions == 0 || repetitions > 50 ||
        warmup_runs > 10) {
      throw std::invalid_argument(
          "Use 1-50 repetitions and 0-10 warm-up runs.");
    }

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    // Both preconditioners use exactly this same frozen linearization.
    const GraphLinearization linearization =
        linearizeGraph(
            initialization.poses,
            dataset.measurements);

    std::cout << std::setprecision(12);

    std::cout
        << "========================================\n"
        << " MATRIX-FREE SCHUR PCG BENCHMARK\n"
        << "========================================\n"
        << "Dataset             : " << dataset_path << '\n'
        << "Poses               : " << initialization.poses.size() << '\n'
        << "Edges               : " << dataset.measurements.size() << '\n'
        << "Trajectory source   : "
        << trajectorySourceName(initialization.source) << '\n'
        << "Initial cost        : " << linearization.cost() << '\n'
        << "Rotation variables  : " << linearization.Jc.cols() << '\n'
        << "Translation variables: "
        << linearization.Jf.cols() - 2 << " (after gauge fixing)\n"
        << "Warm-up runs        : " << warmup_runs << '\n'
        << "Measured repetitions: " << repetitions << '\n'
        << "PCG relative tol    : 1e-10\n"
        << "PCG absolute tol    : 1e-12\n"
        << "PCG max iterations  : 5000\n";

    const std::vector<double> damping_values = {
        1e-3, 1e-1, 1.0
    };

    bool all_passed = true;

    for (double damping : damping_values) {
      const bool passed = runDampingExperiment(
          linearization,
          damping,
          repetitions,
          warmup_runs);

      all_passed = all_passed && passed;
    }

    if (!all_passed) {
      std::cerr << "\nSchur PCG benchmark validation FAILED.\n";
      return EXIT_FAILURE;
    }

    std::cout
        << "\nAll Schur PCG benchmark validation checks PASSED.\n";

    return EXIT_SUCCESS;

  } catch (const std::exception& error) {
    std::cerr
        << "benchmark_schur_pcg FAILED: "
        << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
