#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(const Clock::time_point& begin, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/dataset.g2o\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string filename = argv[1];

    const auto load_begin = Clock::now();
    const G2oDataset dataset = loadG2oSe2(filename);
    const auto load_end = Clock::now();

    const auto reference_begin = Clock::now();
    const PgoEvaluation reference =
        evaluatePgo(dataset.poses, dataset.measurements);
    const auto reference_end = Clock::now();

    const auto symforce_begin = Clock::now();
    const PgoEvaluation generated =
        evaluateSymForcePgo(dataset.poses, dataset.measurements);
    const auto symforce_end = Clock::now();

    const auto linearization_begin = Clock::now();
    const GraphLinearization linearization =
        linearizeGraph(dataset.poses, dataset.measurements);
    const auto linearization_end = Clock::now();

    const double absolute_cost_error =
        std::abs(reference.total - generated.total);
    const double relative_cost_error =
        absolute_cost_error / std::max(1.0, std::abs(reference.total));
    const double residual_cost_error =
        std::abs(linearization.cost() - generated.total);

    const Eigen::Index n = static_cast<Eigen::Index>(dataset.poses.size());
    Eigen::VectorXd vx = Eigen::VectorXd::Zero(2 * n);
    Eigen::VectorXd vy = Eigen::VectorXd::Zero(2 * n);
    for (Eigen::Index i = 0; i < n; ++i) {
      vx(2 * i) = 1.0;
      vy(2 * i + 1) = 1.0;
    }

    std::cout << std::setprecision(15);
    std::cout << "\nDataset                         : " << filename << '\n';
    std::cout << "Poses                           : " << dataset.poses.size() << '\n';
    std::cout << "Edges                           : " << dataset.measurements.size() << '\n';
    std::cout << "VERTEX_SE2 records              : " << dataset.vertex_records << '\n';
    std::cout << "Complete vertex initialization  : "
              << (dataset.has_complete_vertex_initialization ? "yes" : "no")
              << "\n\n";

    std::cout << "Reference PGO cost              : " << reference.total << '\n';
    std::cout << "SymForce PGO cost               : " << generated.total << '\n';
    std::cout << "Absolute cost error             : " << absolute_cost_error << '\n';
    std::cout << "Relative cost error             : " << relative_cost_error << '\n';
    std::cout << "||r||^2 vs SymForce cost error  : " << residual_cost_error << "\n\n";

    std::cout << "Jc shape                        : "
              << linearization.Jc.rows() << " x " << linearization.Jc.cols() << '\n';
    std::cout << "Jf shape                        : "
              << linearization.Jf.rows() << " x " << linearization.Jf.cols() << '\n';
    std::cout << "nnz(Jc)                         : " << linearization.Jc.nonZeros() << '\n';
    std::cout << "nnz(Jf)                         : " << linearization.Jf.nonZeros() << '\n';
    std::cout << "nnz(Hcc)                        : " << linearization.Hcc.nonZeros() << '\n';
    std::cout << "nnz(Hcf)                        : " << linearization.Hcf.nonZeros() << '\n';
    std::cout << "nnz(Hff)                        : " << linearization.Hff.nonZeros() << "\n\n";

    std::cout << "||Jf vx||                       : "
              << (linearization.Jf * vx).norm() << '\n';
    std::cout << "||Jf vy||                       : "
              << (linearization.Jf * vy).norm() << "\n\n";

    std::cout << "Dataset load time [ms]          : "
              << milliseconds(load_begin, load_end) << '\n';
    std::cout << "Handwritten cost time [ms]      : "
              << milliseconds(reference_begin, reference_end) << '\n';
    std::cout << "SymForce cost time [ms]         : "
              << milliseconds(symforce_begin, symforce_end) << '\n';
    std::cout << "Sparse linearization time [ms]  : "
              << milliseconds(linearization_begin, linearization_end) << '\n';

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_g2o FAILED: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
