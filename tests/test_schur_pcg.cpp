#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/schur_pcg.hpp"
#include "cora_symforce/toy_problem.hpp"

namespace {

using DenseMatrix = Eigen::MatrixXd;
using DenseVector = Eigen::VectorXd;

DenseVector solveDenseSchurReference(
    const cora_symforce::GraphLinearization& linearization,
    double damping) {
  const Eigen::Index nc = linearization.Jc.cols();
  const Eigen::Index nf = linearization.Jf.cols() - 2;

  DenseMatrix Hcc(linearization.Hcc);
  DenseMatrix Hcf_full(linearization.Hcf);
  DenseMatrix Hff_full(linearization.Hff);

  DenseMatrix Hcf = Hcf_full.block(0, 2, nc, nf);
  DenseMatrix Hff = Hff_full.block(2, 2, nf, nf);

  Hcc.diagonal().array() += damping;
  Hff.diagonal().array() += damping;

  const DenseVector gc =
      linearization.Jc.transpose() * linearization.residual;

  const DenseVector gf_full =
      linearization.Jf.transpose() * linearization.residual;

  const DenseVector gf = gf_full.tail(nf);

  Eigen::LDLT<DenseMatrix> translation_ldlt;
  translation_ldlt.compute(Hff);

  if (translation_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "Dense reference translation factorization failed.");
  }

  const DenseMatrix X =
      translation_ldlt.solve(Hcf.transpose());

  const DenseVector y =
      translation_ldlt.solve(gf);

  if (translation_ldlt.info() != Eigen::Success ||
      !X.array().isFinite().all() ||
      !y.array().isFinite().all()) {
    throw std::runtime_error(
        "Dense reference translation solve failed.");
  }

  DenseMatrix S = Hcc - Hcf * X;
  S = (0.5 * (S + S.transpose())).eval();

  const DenseVector rhs = -gc + Hcf * y;

  Eigen::LDLT<DenseMatrix> schur_ldlt;
  schur_ldlt.compute(S);

  if (schur_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "Dense reference Schur factorization failed.");
  }

  const DenseVector delta_c =
      schur_ldlt.solve(rhs);

  if (schur_ldlt.info() != Eigen::Success ||
      !delta_c.array().isFinite().all()) {
    throw std::runtime_error(
        "Dense reference Schur solve failed.");
  }

  const DenseVector delta_f =
      translation_ldlt.solve(
          -gf - Hcf.transpose() * delta_c);

  if (translation_ldlt.info() != Eigen::Success ||
      !delta_f.array().isFinite().all()) {
    throw std::runtime_error(
        "Dense reference back-substitution failed.");
  }

  DenseVector result(nc + nf);
  result.head(nc) = delta_c;
  result.tail(nf) = delta_f;
  return result;
}


bool validateDamping(
    const cora_symforce::GraphLinearization& linearization,
    double damping) {
  using namespace cora_symforce;

  const Eigen::Index nc = linearization.Jc.cols();
  const Eigen::Index nf = linearization.Jf.cols() - 2;

  const DenseVector dense =
      solveDenseSchurReference(linearization, damping);

  DenseMatrix Hcc(linearization.Hcc);
  DenseMatrix Hcf_full(linearization.Hcf);
  DenseMatrix Hff_full(linearization.Hff);

  DenseMatrix Hcf = Hcf_full.block(0, 2, nc, nf);
  DenseMatrix Hff = Hff_full.block(2, 2, nf, nf);

  Hcc.diagonal().array() += damping;
  Hff.diagonal().array() += damping;

  const DenseVector gc =
      linearization.Jc.transpose() * linearization.residual;

  const DenseVector gf_full =
      linearization.Jf.transpose() * linearization.residual;

  const DenseVector gf = gf_full.tail(nf);

  DenseMatrix full_H(nc + nf, nc + nf);
  full_H.topLeftCorner(nc, nc) = Hcc;
  full_H.topRightCorner(nc, nf) = Hcf;
  full_H.bottomLeftCorner(nf, nc) = Hcf.transpose();
  full_H.bottomRightCorner(nf, nf) = Hff;

  DenseVector full_g(nc + nf);
  full_g.head(nc) = gc;
  full_g.tail(nf) = gf;

  bool passed = true;

  for (int k = 0; k < 3; ++k) {
    SchurPcgOptions options;
    options.max_iterations = 1000;
    options.relative_tolerance = 1e-12;
    options.absolute_tolerance = 1e-14;
    options.incomplete_cholesky_initial_shift = 1e-3;

    const char* name = nullptr;

    if (k == 0) {
      options.preconditioner =
          SchurPcgPreconditioner::kJacobiHcc;
      name = "P0 Jacobi Hcc";

    } else if (k == 1) {
      options.preconditioner =
          SchurPcgPreconditioner::kSchurBlockJacobiDiagonal;
      name = "P1 Schur block-Jacobi";

    } else {
      options.preconditioner =
          SchurPcgPreconditioner::kSchurIncompleteCholesky;
      name = "P3 Schur incomplete Cholesky";
    }

    const SchurPcgResult pcg =
        solveSchurPcgReducedStep(linearization, damping, options);

    const double solution_gap =
        (pcg.reduced_delta - dense).norm() /
        std::max(1.0, dense.norm());

    const double normal_residual =
        (full_H * pcg.reduced_delta + full_g).norm() /
        std::max(1.0, full_g.norm());

    std::cout
        << "Preconditioner             : " << name << '\n'
        << "Damping                    : " << damping << '\n'
        << "PCG iterations              : " << pcg.iterations << '\n'
        << "PCG true relative residual : "
        << pcg.relative_residual << '\n'
        << "Dense/PCG solution gap      : "
        << solution_gap << '\n'
        << "Full normal relative resid  : "
        << normal_residual << '\n'
        << "Minimum raw diagonal       : "
        << pcg.preconditioner_min_raw_diagonal << '\n'
        << "Floored diagonal entries   : "
        << pcg.preconditioner_floored_diagonal_entries
        << " / " << nc << "\n\n";

    bool current_passed = true;

    if (!pcg.reduced_delta.array().isFinite().all() ||
        !std::isfinite(pcg.preconditioner_min_raw_diagonal)) {
      std::cerr << "FAILED: non-finite P0/P1 diagnostic or solution.\n";
      current_passed = false;
    }

    if (!std::isfinite(solution_gap) || solution_gap > 1e-9) {
      std::cerr << "FAILED: disagreement with dense Schur reference.\n";
      current_passed = false;
    }

    if (!std::isfinite(normal_residual) ||
        normal_residual > 1e-10) {
      std::cerr << "FAILED: excessive full normal-equation residual.\n";
      current_passed = false;
    }

    if (!std::isfinite(pcg.relative_residual) ||
        pcg.relative_residual > 1e-10) {
      std::cerr << "FAILED: PCG did not meet residual tolerance.\n";
      current_passed = false;
    }

    if (pcg.preconditioner_floored_diagonal_entries >
        static_cast<std::size_t>(nc)) {
      std::cerr << "FAILED: invalid number of floored entries.\n";
      current_passed = false;
    }

    std::cout << "Result                      : "
              << (current_passed ? "PASSED" : "FAILED")
              << "\n\n";

    passed = passed && current_passed;
  }

  return passed;
}

}  // namespace

int main() {
  using namespace cora_symforce;

  const ToyProblem problem = makeToyProblem();
  const GraphLinearization linearization =
      linearizeGraph(
          problem.initial_guess,
          problem.measurements);

  bool passed = true;
  passed &= validateDamping(linearization, 1e-3);
  passed &= validateDamping(linearization, 1e-1);

  if (!passed) {
    return EXIT_FAILURE;
  }

  std::cout << "Matrix-free Schur PCG checks PASSED.\n";
  return EXIT_SUCCESS;
}
