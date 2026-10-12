#include "cora_symforce/symforce_pgo_cost.hpp"

#include <cmath>
#include <stdexcept>

#include "edge_jacobian.h"
#include "edge_residual.h"

namespace cora_symforce {

namespace {

double rotationAngle(const Eigen::Matrix2d& R)
{
    return std::atan2(R(1, 0), R(0, 0));
}


void validateMeasurementIndices(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement)
{
    if (measurement.i >= poses.size() ||
        measurement.j >= poses.size()) {
        throw std::out_of_range(
            "RelativePoseMeasurement references an invalid pose index.");
    }
}


Eigen::Matrix<double, 6, 1> evaluateGeneratedResidual(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement)
{
    validateMeasurementIndices(poses, measurement);

    const Pose2State& pose_i = poses[measurement.i];
    const Pose2State& pose_j = poses[measurement.j];

    const double theta_i = rotationAngle(pose_i.R);
    const double theta_j = rotationAngle(pose_j.R);

    return cora_symforce_generated::EdgeResidual<double>(
        theta_i,
        pose_i.t,
        theta_j,
        pose_j.t,
        measurement.R,
        measurement.t,
        measurement.kappa,
        measurement.tau);
}

}  // namespace


EdgeCost evaluateSymForceEdge(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement)
{
    const Eigen::Matrix<double, 6, 1> residual =
        evaluateGeneratedResidual(
            poses,
            measurement);

    EdgeCost edge{};

    edge.i = measurement.i;
    edge.j = measurement.j;

    /*
     * First four entries are vec(rotation residual).
     * Last two entries are translation residual.
     *
     * The generated residual already contains sqrt(kappa)
     * and sqrt(tau), so squaredNorm() gives the weighted cost.
     */
    edge.rotation =
        residual.head<4>().squaredNorm();

    edge.translation =
        residual.tail<2>().squaredNorm();

    return edge;
}


PgoEvaluation evaluateSymForcePgo(
    const std::vector<Pose2State>& poses,
    const std::vector<RelativePoseMeasurement>& measurements)
{
    PgoEvaluation evaluation{};

    evaluation.total = 0.0;
    evaluation.edges.reserve(measurements.size());

    for (const auto& measurement : measurements) {
        const EdgeCost edge =
            evaluateSymForceEdge(
                poses,
                measurement);

        evaluation.total += edge.total();
        evaluation.edges.push_back(edge);
    }

    return evaluation;
}


SymForceEdgeJacobian evaluateSymForceEdgeJacobian(
    const std::vector<Pose2State>& poses,
    const RelativePoseMeasurement& measurement)
{
    validateMeasurementIndices(
        poses,
        measurement);

    const Pose2State& pose_i =
        poses[measurement.i];

    const Pose2State& pose_j =
        poses[measurement.j];

    const double theta_i =
        rotationAngle(pose_i.R);

    const double theta_j =
        rotationAngle(pose_j.R);

    return cora_symforce_generated::EdgeJacobian<double>(
        theta_i,
        pose_i.t,
        theta_j,
        pose_j.t,
        measurement.R,
        measurement.t,
        measurement.kappa,
        measurement.tau);
}

}  // namespace cora_symforce
