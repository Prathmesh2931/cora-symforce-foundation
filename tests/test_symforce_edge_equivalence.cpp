#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "edge_jacobian.h"
#include "edge_residual.h"

namespace {

using Vector2 = Eigen::Vector2d;
using Matrix2 = Eigen::Matrix2d;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;


Matrix2 rotationMatrix(const double theta)
{
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    Matrix2 R;
    R << c, -s,
         s,  c;

    return R;
}


/*
 * Direct handwritten residual.
 *
 * This intentionally duplicates the mathematical definition rather than
 * calling the generated code. It is the independent numerical reference
 * against which the SymForce output is checked.
 */
Vector6 referenceResidual(
    const double theta_i,
    const Vector2& t_i,
    const double theta_j,
    const Vector2& t_j,
    const Matrix2& measured_R_ij,
    const Vector2& measured_t_ij,
    const double kappa,
    const double tau)
{
    const Matrix2 R_i = rotationMatrix(theta_i);
    const Matrix2 R_j = rotationMatrix(theta_j);

    const Matrix2 rotation_error =
        std::sqrt(kappa) *
        (R_j - R_i * measured_R_ij);

    const Vector2 translation_error =
        std::sqrt(tau) *
        (t_j - t_i - R_i * measured_t_ij);

    Vector6 residual;

    // Column-major vec(), matching Eigen and the SE-Sync B3 convention.
    residual <<
        rotation_error(0, 0),
        rotation_error(1, 0),
        rotation_error(0, 1),
        rotation_error(1, 1),
        translation_error(0),
        translation_error(1);

    return residual;
}


Vector6 generatedResidual(
    const Vector6& state,
    const Matrix2& measured_R_ij,
    const Vector2& measured_t_ij,
    const double kappa,
    const double tau)
{
    const double theta_i = state(0);

    Vector2 t_i;
    t_i << state(1), state(2);

    const double theta_j = state(3);

    Vector2 t_j;
    t_j << state(4), state(5);

    return cora_symforce_generated::EdgeResidual<double>(
        theta_i,
        t_i,
        theta_j,
        t_j,
        measured_R_ij,
        measured_t_ij,
        kappa,
        tau);
}


Matrix6 finiteDifferenceJacobian(
    const Vector6& state,
    const Matrix2& measured_R_ij,
    const Vector2& measured_t_ij,
    const double kappa,
    const double tau)
{
    constexpr double epsilon = 1e-6;

    Matrix6 J;

    for (Eigen::Index k = 0; k < state.size(); ++k) {
        Vector6 plus = state;
        Vector6 minus = state;

        plus(k) += epsilon;
        minus(k) -= epsilon;

        const Vector6 r_plus = generatedResidual(
            plus,
            measured_R_ij,
            measured_t_ij,
            kappa,
            tau);

        const Vector6 r_minus = generatedResidual(
            minus,
            measured_R_ij,
            measured_t_ij,
            kappa,
            tau);

        J.col(k) =
            (r_plus - r_minus) / (2.0 * epsilon);
    }

    return J;
}


bool checkNear(
    const double error,
    const double tolerance,
    const char* name)
{
    std::cout
        << std::left
        << std::setw(34)
        << name
        << " error = "
        << std::scientific
        << error
        << '\n';

    if (error > tolerance) {
        std::cerr
            << "FAILED: "
            << name
            << " exceeded tolerance "
            << tolerance
            << '\n';

        return false;
    }

    return true;
}

}  // namespace


int main()
{
    /*
     * Deterministic non-trivial state.
     *
     * These numbers are intentionally not a zero-residual configuration.
     * That prevents a broken implementation from passing simply because
     * every term evaluates to zero.
     */
    const double theta_i = 0.23;
    const double theta_j = 0.47;

    Vector2 t_i;
    t_i << 0.31, -0.18;

    Vector2 t_j;
    t_j << 1.27, 0.22;

    /*
     * Construct a valid relative-rotation measurement and perturb it
     * slightly from the current predicted relative rotation.
     */
    const double measured_theta_ij = 0.21;
    const Matrix2 measured_R_ij =
        rotationMatrix(measured_theta_ij);

    Vector2 measured_t_ij;
    measured_t_ij << 0.93, 0.16;

    const double kappa = 4.7;
    const double tau = 3.2;


    Vector6 state;
    state <<
        theta_i,
        t_i.x(),
        t_i.y(),
        theta_j,
        t_j.x(),
        t_j.y();


    // Residual equivalence
    const Vector6 reference = referenceResidual(
        theta_i,
        t_i,
        theta_j,
        t_j,
        measured_R_ij,
        measured_t_ij,
        kappa,
        tau);

    const Vector6 generated = generatedResidual(
        state,
        measured_R_ij,
        measured_t_ij,
        kappa,
        tau);

    const double residual_error =
        (reference - generated).norm();


    // Cost equivalence
    const double reference_cost =
        reference.squaredNorm();

    const double generated_cost =
        generated.squaredNorm();

    const double cost_error =
        std::abs(reference_cost - generated_cost);


    // Analytic generated Jacobian
    const Matrix6 generated_jacobian =
        cora_symforce_generated::EdgeJacobian<double>(
            theta_i,
            t_i,
            theta_j,
            t_j,
            measured_R_ij,
            measured_t_ij,
            kappa,
            tau);


    // Independent finite-difference Jacobian
    const Matrix6 finite_difference_jacobian =
        finiteDifferenceJacobian(
            state,
            measured_R_ij,
            measured_t_ij,
            kappa,
            tau);

    const double jacobian_error =
        (generated_jacobian -
         finite_difference_jacobian)
            .cwiseAbs()
            .maxCoeff();


    // Report
    std::cout << std::setprecision(12);

    std::cout << "\nReference residual:\n"
              << reference.transpose()
              << "\n\nGenerated residual:\n"
              << generated.transpose()
              << "\n\n";

    std::cout
        << "Reference cost : "
        << reference_cost
        << '\n';

    std::cout
        << "Generated cost : "
        << generated_cost
        << "\n\n";


    bool passed = true;

    passed &= checkNear(
        residual_error,
        1e-12,
        "Residual equivalence");

    passed &= checkNear(
        cost_error,
        1e-12,
        "Cost equivalence");

    passed &= checkNear(
        jacobian_error,
        1e-6,
        "Jacobian finite difference");


    if (!passed) {
        std::cerr
            << "\nSymForce equivalence test FAILED.\n";

        return EXIT_FAILURE;
    }


    std::cout
        << "\nSymForce residual/cost/Jacobian checks PASSED.\n";

    return EXIT_SUCCESS;
}
