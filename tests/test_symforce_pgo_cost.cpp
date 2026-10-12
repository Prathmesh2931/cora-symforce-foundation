#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cora_symforce/measurement.hpp"
#include "cora_symforce/pgo_cost.hpp"
#include "cora_symforce/pose2.hpp"
#include "cora_symforce/symforce_pgo_cost.hpp"

namespace {

Eigen::Matrix2d rotationMatrix(const double theta)
{
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;

    return R;
}


cora_symforce::Pose2State makePose(
    const double theta,
    const double x,
    const double y)
{
    cora_symforce::Pose2State pose;

    pose.R = rotationMatrix(theta);
    pose.t << x, y;

    return pose;
}


cora_symforce::RelativePoseMeasurement makeMeasurement(
    const std::size_t i,
    const std::size_t j,
    const double measured_theta,
    const double tx,
    const double ty,
    const double kappa,
    const double tau)
{
    cora_symforce::RelativePoseMeasurement measurement;

    measurement.i = i;
    measurement.j = j;

    measurement.R =
        rotationMatrix(measured_theta);

    measurement.t << tx, ty;

    measurement.kappa = kappa;
    measurement.tau = tau;

    return measurement;
}


bool checkNear(
    const double error,
    const double tolerance,
    const char* label)
{
    std::cout
        << std::left
        << std::setw(40)
        << label
        << " error = "
        << std::scientific
        << error
        << '\n';

    if (error > tolerance) {
        std::cerr
            << "FAILED: "
            << label
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
    using namespace cora_symforce;

    /*
     * Three non-trivial poses.
     *
     * We intentionally do not construct a zero-residual problem.
     * Both implementations must evaluate the same non-zero objective.
     */
    std::vector<Pose2State> poses;

    poses.push_back(
        makePose(
            0.05,
            0.00,
            0.00));

    poses.push_back(
        makePose(
            0.24,
            1.02,
            0.17));

    poses.push_back(
        makePose(
            0.41,
            2.03,
            0.11));


    /*
     * Three edges:
     *
     *   0 -> 1
     *   1 -> 2
     *   0 -> 2
     *
     * The loop-closure edge is useful because this is now genuinely
     * a graph-level comparison rather than just repeating a single edge.
     */
    std::vector<RelativePoseMeasurement> measurements;

    measurements.push_back(
        makeMeasurement(
            0,
            1,
            0.18,
            0.98,
            0.12,
            4.5,
            3.1));

    measurements.push_back(
        makeMeasurement(
            1,
            2,
            0.20,
            1.01,
            -0.03,
            5.0,
            2.8));

    measurements.push_back(
        makeMeasurement(
            0,
            2,
            0.39,
            2.01,
            0.08,
            3.7,
            2.4));


    // Existing handwritten PGO implementation

    const PgoEvaluation reference =
        evaluatePgo(
            poses,
            measurements);


    // SymForce-generated PGO implementation
    const PgoEvaluation generated =
        evaluateSymForcePgo(
            poses,
            measurements);


    std::cout << std::setprecision(15);

    std::cout
        << "\nReference full PGO cost : "
        << reference.total
        << '\n';

    std::cout
        << "SymForce full PGO cost  : "
        << generated.total
        << "\n\n";


    bool passed = true;

    // Full graph cost
    passed &= checkNear(
        std::abs(
            reference.total -
            generated.total),
        1e-12,
        "Full graph cost equivalence");

    // Edge count
    if (reference.edges.size() !=
        generated.edges.size()) {

        std::cerr
            << "FAILED: edge count mismatch.\n";

        return EXIT_FAILURE;
    }


    // Per-edge decomposition
    for (std::size_t k = 0;
         k < reference.edges.size();
         ++k) {

        const EdgeCost& ref =
            reference.edges[k];

        const EdgeCost& gen =
            generated.edges[k];


        std::cout
            << "\nEdge "
            << ref.i
            << " -> "
            << ref.j
            << '\n';


        passed &= checkNear(
            std::abs(
                ref.rotation -
                gen.rotation),
            1e-12,
            "  rotation cost");


        passed &= checkNear(
            std::abs(
                ref.translation -
                gen.translation),
            1e-12,
            "  translation cost");


        passed &= checkNear(
            std::abs(
                ref.total() -
                gen.total()),
            1e-12,
            "  total edge cost");
    }


    if (!passed) {

        std::cerr
            << "\nGraph-level SymForce PGO test FAILED.\n";

        return EXIT_FAILURE;
    }


    std::cout
        << "\nGraph-level SymForce PGO equivalence PASSED.\n";

    return EXIT_SUCCESS;
}
