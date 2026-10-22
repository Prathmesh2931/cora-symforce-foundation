#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "cora_symforce/g2o_dataset.hpp"
#include "cora_symforce/graph_linearization.hpp"
#include "cora_symforce/symmetric_solver.hpp"
#include "cora_symforce/trajectory_io.hpp"

int main(int argc, char** argv) {
  using namespace cora_symforce;

  if (argc != 2 && argc != 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " path/to/dataset.g2o [repetitions]\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string dataset_path = argv[1];

    LinearSolverComparisonOptions options;
    if (argc == 3) {
      options.repetitions =
          static_cast<std::size_t>(std::stoul(argv[2]));
    }

    const G2oDataset dataset = loadG2oSe2(dataset_path);
    const TrajectoryInitialization initialization =
        initializeTrajectory(dataset);

    // Important: both solvers see exactly this same frozen linearization.
    const GraphLinearization linearization =
        linearizeGraph(initialization.poses, dataset.measurements);

    const LinearSolverComparison comparison =
        compareLinearSolvers(linearization, options);

    const double qr_total_ms =
        comparison.qr_factorization_time_ms +
        comparison.qr_solve_time_ms;

    const double symmetric_total_ms =
        comparison.normal_formation_time_ms +
        comparison.ldlt_factorization_time_ms +
        comparison.ldlt_solve_time_ms;

    std::cout << std::setprecision(15);
    std::cout << "\nDataset                           : "
              << dataset_path << '\n';
    std::cout << "Poses                             : "
              << initialization.poses.size() << '\n';
    std::cout << "Edges                             : "
              << dataset.measurements.size() << '\n';
    std::cout << "Trajectory source                 : "
              << trajectorySourceName(initialization.source) << '\n';
    std::cout << "Frozen linearization cost         : "
              << linearization.cost() << '\n';
    std::cout << "Timing repetitions                : "
              << options.repetitions << "\n\n";

    std::cout << "J shape                           : "
              << comparison.residual_rows << " x "
              << comparison.state_columns << '\n';
    std::cout << "J nonzeros                        : "
              << comparison.jacobian_nonzeros << '\n';
    std::cout << "SparseQR numerical rank           : "
              << comparison.qr_rank << '\n';
    std::cout << "Gauge-fixed translation columns   : "
              << comparison.fixed_tx_column << ", "
              << comparison.fixed_ty_column << '\n';
    std::cout << "Reduced H shape                   : "
              << comparison.reduced_state_columns << " x "
              << comparison.reduced_state_columns << '\n';
    std::cout << "H nonzeros                        : "
              << comparison.normal_nonzeros << '\n';
    std::cout << "H symmetry max error              : "
              << comparison.normal_symmetry_error << '\n';

    std::cout << "Schur rotation columns            : "
              << comparison.schur_rotation_columns << '\n';

    std::cout << "Schur translation columns         : "
              << comparison.schur_translation_columns << '\n';

    std::cout << "Schur symmetry max error          : "
              << comparison.schur_symmetry_error << "\n\n";

    std::cout << "SparseQR factor median [ms]        : "
              << comparison.qr_factorization_time_ms << '\n';
    std::cout << "SparseQR solve median [ms]         : "
              << comparison.qr_solve_time_ms << '\n';
    std::cout << "SparseQR total median [ms]         : "
              << qr_total_ms << "\n\n";

    std::cout << "Normal matrix formation [ms]       : "
              << comparison.normal_formation_time_ms << '\n';
    std::cout << "SimplicialLDLT factor median [ms]  : "
              << comparison.ldlt_factorization_time_ms << '\n';
    std::cout << "SimplicialLDLT solve median [ms]   : "
              << comparison.ldlt_solve_time_ms << '\n';
    std::cout << "Symmetric path total [ms]          : "
              << symmetric_total_ms << '\n';

    if (symmetric_total_ms > 0.0) {
      std::cout << "QR / symmetric time ratio          : "
                << qr_total_ms / symmetric_total_ms << "\n\n";
    } else {
      std::cout << "QR / symmetric time ratio          : inf\n\n";
    }

    std::cout << "||J dx_qr + r||                    : "
              << comparison.qr_linearized_residual_norm << '\n';
    std::cout << "||J dx_ldlt + r||                  : "
              << comparison.ldlt_linearized_residual_norm << '\n';
    std::cout << "Linearized residual norm gap       : "
              << comparison.linearized_residual_norm_gap << '\n';
    std::cout << "||J(dx_qr - dx_ldlt)||             : "
              << comparison.prediction_difference_norm << '\n';
    std::cout << "Relative ||H dx_ldlt + g||         : "
              << comparison.ldlt_relative_normal_residual << '\n';

    std::cout << "||J dx_schur + r||                 : "
              << comparison.schur_linearized_residual_norm << '\n';

    std::cout << "Schur vs LDLT residual norm gap    : "
              << std::abs(
                     comparison.schur_linearized_residual_norm -
                     comparison.ldlt_linearized_residual_norm)
              << '\n';

    std::cout << "||J(dx_ldlt - dx_schur)||          : "
              << comparison.schur_prediction_difference_norm << '\n';

    std::cout << "Relative ||H dx_schur + g||        : "
              << comparison.schur_relative_normal_residual << '\n';

    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr
        << "benchmark_linear_solvers FAILED: "
        << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
