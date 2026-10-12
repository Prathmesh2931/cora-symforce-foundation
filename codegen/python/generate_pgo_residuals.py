#!/usr/bin/env python3

from pathlib import Path

import symforce

# Use SymForce's safe numerical epsilon configuration.
# The current SE-Sync residual does not contain singular operations,
# but configuring epsilon explicitly avoids relying on the global default.
symforce.set_epsilon_to_number()

from symforce import codegen
from symforce import symbolic as sf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "generated"
NAMESPACE = "cora_symforce_generated"


def rot2(theta: sf.Scalar):
    """
    2D rotation matrix.

    We keep theta as the minimal rotation variable so the generated
    Jacobian is with respect to the physical 1-DOF SO(2) rotation,
    rather than treating the four matrix entries as independent.
    """
    c = sf.cos(theta)
    s = sf.sin(theta)

    return sf.Matrix(
        [
            [c, -s],
            [s,  c],
        ]
    )


def edge_residual(
    theta_i: sf.Scalar,
    t_i: sf.Matrix21,
    theta_j: sf.Scalar,
    t_j: sf.Matrix21,
    measured_R_ij: sf.Matrix22,
    measured_t_ij: sf.Matrix21,
    kappa: sf.Scalar,
    tau: sf.Scalar,
):
    """
    Square-root weighted SE-Sync relative-pose residual.

    Rotation:
        sqrt(kappa) * (R_j - R_i * R~_ij)

    Translation:
        sqrt(tau) * (t_j - t_i - R_i * t~_ij)

    The 2x2 rotation residual is vectorized column-major so it
    matches Eigen's default storage and the SE-Sync B3 convention.

    Output ordering:
        [ eR(0,0),
          eR(1,0),
          eR(0,1),
          eR(1,1),
          et(0),
          et(1) ]
    """
    R_i = rot2(theta_i)
    R_j = rot2(theta_j)

    rotation_error = sf.sqrt(kappa) * (
        R_j - R_i * measured_R_ij
    )

    translation_error = sf.sqrt(tau) * (
        t_j - t_i - R_i * measured_t_ij
    )

    return sf.Matrix(
        [
            rotation_error[0, 0],
            rotation_error[1, 0],
            rotation_error[0, 1],
            rotation_error[1, 1],
            translation_error[0],
            translation_error[1],
        ]
    )


def edge_jacobian(
    theta_i: sf.Scalar,
    t_i: sf.Matrix21,
    theta_j: sf.Scalar,
    t_j: sf.Matrix21,
    measured_R_ij: sf.Matrix22,
    measured_t_ij: sf.Matrix21,
    kappa: sf.Scalar,
    tau: sf.Scalar,
):
    """
    Jacobian of the SE-Sync residual with respect to

        x = [theta_i, t_ix, t_iy, theta_j, t_jx, t_jy].

    Shape:
        6 residual components x 6 state perturbations.
    """
    residual = edge_residual(
        theta_i,
        t_i,
        theta_j,
        t_j,
        measured_R_ij,
        measured_t_ij,
        kappa,
        tau,
    )

    state = sf.Matrix(
        [
            theta_i,
            t_i[0],
            t_i[1],
            theta_j,
            t_j[0],
            t_j[1],
        ]
    )

    return residual.jacobian(state)


def generate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    residual_codegen = codegen.Codegen.function(
        func=edge_residual,
        name="edge_residual",
        config=codegen.CppConfig(),
        output_names=["res"],
        return_key="res",
    )

    jacobian_codegen = codegen.Codegen.function(
        func=edge_jacobian,
        name="edge_jacobian",
        config=codegen.CppConfig(),
        output_names=["res"],
        return_key="res",
    )

    residual_data = residual_codegen.generate_function(
        output_dir=OUTPUT_DIR,
        namespace=NAMESPACE,
        skip_directory_nesting=True,
    )

    jacobian_data = jacobian_codegen.generate_function(
        output_dir=OUTPUT_DIR,
        namespace=NAMESPACE,
        skip_directory_nesting=True,
    )

    print("Generated SymForce C++ files:")

    for file_path in residual_data.generated_files:
        print(f"  {file_path}")

    for file_path in jacobian_data.generated_files:
        print(f"  {file_path}")


if __name__ == "__main__":
    generate()
