#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"

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

}  // namespace

int main() {
  using namespace cora_symforce;

  const std::string with_vertices = "test_se2_with_vertices.g2o";
  const std::string edge_only = "test_se2_edge_only.g2o";

  {
    std::ofstream out(with_vertices);
    out << "VERTEX_SE2 0 0.0 0.0 0.0\n";
    out << "VERTEX_SE2 1 1.1 0.2 0.1\n";
    out << "VERTEX_SE2 2 2.0 0.1 0.25\n";
    out << "EDGE_SE2 0 1 1.0 0.1 0.08 4.0 1.0 0.0 3.0 0.0 5.0\n";
    out << "EDGE_SE2 1 2 0.9 -0.1 0.14 6.0 0.5 0.0 4.0 0.0 7.0\n";
    out << "EDGE_SE2 0 2 2.0 0.0 0.24 5.0 0.2 0.0 4.5 0.0 6.0\n";
  }

  {
    std::ofstream out(edge_only);
    out << "EDGE_SE2 0 1 1.0 0.0 0.10 4.0 0.0 0.0 4.0 0.0 5.0\n";
    out << "EDGE_SE2 1 2 1.0 0.0 0.10 4.0 0.0 0.0 4.0 0.0 5.0\n";
  }

  bool passed = true;

  try {
    const double expected_tau = 22.0 / 7.0;
    const double tau = seSyncTauFromTranslationInformation(4.0, 1.0, 3.0);
    passed &= checkNear(
        std::abs(tau - expected_tau),
        1e-12,
        "SE-Sync tau conversion");

    const G2oDataset dataset = loadG2oSe2(with_vertices);
    if (dataset.poses.size() != 3 ||
        dataset.measurements.size() != 3 ||
        !dataset.has_complete_vertex_initialization) {
      std::cerr << "FAILED: vertex-backed dataset metadata mismatch.\n";
      passed = false;
    }

    passed &= checkNear(
        std::abs(dataset.measurements.front().kappa - 5.0),
        1e-12,
        "SE-Sync kappa conversion");

    const PgoEvaluation reference =
        evaluatePgo(dataset.poses, dataset.measurements);
    const PgoEvaluation generated =
        evaluateSymForcePgo(dataset.poses, dataset.measurements);
    const GraphLinearization linearization =
        linearizeGraph(dataset.poses, dataset.measurements);

    passed &= checkNear(
        std::abs(reference.total - generated.total),
        1e-11,
        "Parsed graph handwritten/SymForce cost");
    passed &= checkNear(
        std::abs(linearization.cost() - generated.total),
        1e-11,
        "Parsed graph residual-stack cost");

    const G2oDataset no_vertices = loadG2oSe2(edge_only);
    if (no_vertices.poses.size() != 3 ||
        no_vertices.measurements.size() != 2 ||
        no_vertices.has_complete_vertex_initialization) {
      std::cerr << "FAILED: edge-only dataset fallback mismatch.\n";
      passed = false;
    }

    const PgoEvaluation edge_reference =
        evaluatePgo(no_vertices.poses, no_vertices.measurements);
    const PgoEvaluation edge_generated =
        evaluateSymForcePgo(no_vertices.poses, no_vertices.measurements);

    passed &= checkNear(
        std::abs(edge_reference.total - edge_generated.total),
        1e-11,
        "Edge-only graph handwritten/SymForce cost");
  } catch (const std::exception& error) {
    std::cerr << "FAILED with exception: " << error.what() << '\n';
    passed = false;
  }

  std::remove(with_vertices.c_str());
  std::remove(edge_only.c_str());

  if (!passed) {
    std::cerr << "\ng2o dataset test FAILED.\n";
    return EXIT_FAILURE;
  }

  std::cout << "\ng2o dataset checks PASSED.\n";
  return EXIT_SUCCESS;
}
