# CORA / SE-Sync SymForce Foundation

A C++ research implementation for reproducing and validating the
SE-Sync pose-graph optimization objective using **SymForce-generated
residuals and Jacobians**, sparse graph assembly, and nonlinear solver
benchmarking.

The current work focuses on **SE(2) pose-graph optimization** and provides a
validated numerical foundation for studying:

- exact SE-Sync formulations,
- analytical translation elimination,
- SymForce residual and Jacobian generation,
- sparse Gauss-Newton optimization,
- SparseQR and symmetric LDLT linear solvers,
- initialization sensitivity,
- and future certifiable / structured optimization methods.

> This repository does not currently implement the complete SE-Sync global
> solver or CORA. The goal is to validate each mathematical and numerical
> component independently before extending the solver.

---

## Current Pipeline

```text
Relative SE(2) measurements
          |
          v
Direct SE-Sync objective
          |
          +----------------------+
          |                      |
          v                      v
   B1 / B2 / B3             SymForce residual
          |                      |
          v                      v
Explicit quadratic M       Generated Jacobian
          |                      |
          v                      v
Translation elimination    Sparse graph assembly
          |                      |
          v                      v
Reduced rotation Q          Gauss-Newton
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
                SparseQR              SimplicialLDLT
                    |                         |
                    +------------+------------+
                                 |
                                 v
                       Solver validation
                                 |
                                 v
                     Official SE-Sync check
```

---

# Key Results

## SymForce Residual and Jacobian Validation

The generated SymForce implementation reproduces the handwritten SE-Sync
objective to numerical precision.

```text
Residual equivalence error       = 2.002685474353e-16
Cost equivalence error           = 5.551115123126e-17
Jacobian finite-difference error = 1.654302250742e-10

2 J^T r vs FD cost gradient      ≈ 4.79e-11
J v vs FD residual derivative    ≈ 9.51e-11
```

---

## SparseQR vs Symmetric LDLT

Both nonlinear solvers use exactly the same:

- objective,
- initialization,
- residuals,
- Jacobians,
- line search,
- stopping conditions,
- and state updates.

Only the linear solver changes.

### SparseQR reference

The reference backend solves:

$$
J \Delta x \approx -r
$$

using Eigen `SparseQR`.

### Symmetric LDLT backend

The second backend solves the gauge-reduced normal equations:

$$
H_r \Delta x_r = -g_r
$$

with

$$
H_r = J_r^T J_r
$$

and

$$
g_r = J_r^T r
$$

using Eigen `SimplicialLDLT`.

The first-pose translation increments are fixed to remove the exact global
translation gauge.

---

# Solver Validation Figures

The validation plots compare:

- objective convergence,
- cumulative solver runtime,
- final aligned trajectories,
- position error,
- and heading error.

<table>
<tr>
<th align="center">MIT</th>
<th align="center">CSAIL</th>
</tr>
<tr>
<td align="center">
<img src="docs/figures/mit_solver_validation.png"
     alt="MIT solver validation"
     width="100%">
</td>
<td align="center">
<img src="docs/figures/csail_solver_validation.png"
     alt="CSAIL solver validation"
     width="100%">
</td>
</tr>
</table>

These plots validate **solver equivalence**. SparseQR is used as an
independent numerical reference, not as a global-optimality reference.

---

# Dataset Results

## CSAIL

The CSAIL graph contains:

```text
poses = 1045
edges = 1172
```

Both SparseQR and LDLT converge in four nonlinear iterations.

```text
QR final cost       = 31.7037158836249
LDLT final cost     = 31.7037158836249

iterations          = 4
relative cost gap   ≈ 2.24e-15
```

Representative three-run median timings:

```text
SparseQR solve time ≈ 18632.565 ms
LDLT solve time     ≈     5.381 ms

SparseQR wall time  ≈ 18635.963 ms
LDLT wall time      ≈     8.605 ms
```

Measured ratios:

```text
linear-solve ratio ≈ 3463x
wall-time ratio    ≈ 2166x
```

These timings apply only to the current implementation and experimental
configuration. They are not intended as a universal performance comparison
between QR and LDLT.

---

## MIT

The MIT graph contains:

```text
poses = 808
edges = 827
```

Starting from the supplied `VERTEX_SE2` trajectory, both nonlinear backends
converge to approximately:

```text
F = 1298.03
```

Extending the maximum iteration count from 30 to 200 gives:

```text
QR final cost      = 1298.03279183363
LDLT final cost    = 1298.03279183363

QR iterations      = 69
LDLT iterations    = 69

QR solve total     ≈ 147450.19 ms
LDLT solve total   ≈     61.90 ms
```

The aligned final trajectories also agree numerically:

```text
ATE RMSE             ≈ 5.99e-13 m
maximum position gap ≈ 1.62e-12 m
heading RMSE         ≈ 8.87e-15 rad
```

Therefore, the MIT result is not caused by:

- the previous 30-iteration limit,
- disagreement between QR and LDLT,
- or an obvious error in the local linear solve.

The remaining issue is the nonconvex optimization basin.

---

# SE-Sync Pose-Graph Objective

For an edge between poses $i$ and $j$, let

- $R_i$ and $R_j$ be absolute rotations,
- $t_i$ and $t_j$ be absolute translations,
- $\widetilde{R}_{ij}$ be the measured relative rotation,
- $\widetilde{t}_{ij}$ be the measured relative translation,
- $\kappa_{ij}$ be the rotational precision,
- $\tau_{ij}$ be the translational precision.

The SE-Sync objective implemented in this repository is:

$$
F(R,t)
=
\sum_{(i,j)}
\kappa_{ij}
\left\|
R_j - R_i \widetilde{R}_{ij}
\right\|_F^2
+
\tau_{ij}
\left\|
t_j - t_i - R_i \widetilde{t}_{ij}
\right\|_2^2
$$

---

# g2o Precision Conversion

For an `EDGE_SE2` measurement, define the translational information block:

$$
\Omega_{t,ij}
=
\begin{bmatrix}
I_{11} & I_{12} \\
I_{12} & I_{22}
\end{bmatrix}
$$

The isotropic SE-Sync translation precision is:

$$
\tau_{ij}
=
\frac{2}
{\mathrm{tr}\left(\Omega_{t,ij}^{-1}\right)}
$$

For the symmetric two-dimensional block, this is equivalent to:

$$
\tau_{ij}
=
\frac{
2 \det(\Omega_{t,ij})
}{
\mathrm{tr}(\Omega_{t,ij})
}
$$

The rotational precision is:

$$
\kappa_{ij}=I_{33}
$$

Translation/rotation cross-information entries are intentionally not included
because this implementation follows the SE-Sync isotropic precision model.

---

# Equivalent SE-Sync Formulations

## Direct Residuals

The rotational residual is:

$$
e^R_{ij}
=
R_j - R_i \widetilde{R}_{ij}
$$

The translational residual is:

$$
e^t_{ij}
=
t_j - t_i - R_i \widetilde{t}_{ij}
$$

The weighted edge cost is:

$$
F_{ij}
=
\kappa_{ij}
\left\|e^R_{ij}\right\|_F^2
+
\tau_{ij}
\left\|e^t_{ij}\right\|_2^2
$$

The handwritten implementation in:

```text
src/pgo_cost.cpp
```

is retained as an independent numerical reference.

---

## B1 / B2 / B3 Residual Representation

The same objective can be represented using linear residual operators.

The stacked translation residual is:

$$
e_t = B_1 t + B_2 r
$$

and the stacked rotational residual is:

$$
e_R = B_3 r
$$

where $r$ contains the column-major vectorization of all rotation matrices.

The complete objective becomes:

$$
F
=
\left\|
B_1 t + B_2 r
\right\|_2^2
+
\left\|
B_3 r
\right\|_2^2
$$

At the edge level:

$$
B_1 t
=
\sqrt{\tau_{ij}}
(t_j-t_i)
$$

$$
B_2 r
=
-\sqrt{\tau_{ij}}
R_i \widetilde{t}_{ij}
$$

and:

$$
B_3 r
=
\sqrt{\kappa_{ij}}
\mathrm{vec}
\left(
R_j-R_i\widetilde{R}_{ij}
\right)
$$

The matrix vectorization convention is column-major:

$$
\mathrm{vec}(R)
=
\begin{bmatrix}
R_{00} \\
R_{10} \\
R_{01} \\
R_{11}
\end{bmatrix}
$$

The implementation verifies numerically that the direct and operator
representations produce the same residuals and objective.

---

# Explicit Quadratic Formulation

The same PGO objective can be written as:

$$
F(Y)
=
\mathrm{tr}
\left(
Y M Y^T
\right)
$$

where the quadratic data matrix has the block structure:

$$
M
=
\begin{bmatrix}
L_{\tau} & V \\
V^T & L_{\rho}+\Sigma
\end{bmatrix}
$$

where:

- $L_{\tau}$ is the weighted translation graph Laplacian,
- $V$ contains translation-rotation coupling,
- $\Sigma$ contains translation-measurement quadratic terms,
- $L_{\rho}$ is the rotational connection Laplacian.

This is an algebraically equivalent representation of the original
pose-graph objective.

No relaxation has been introduced at this stage.

---

# Exact Translation Elimination

For fixed rotations, translation estimation is a linear least-squares problem:

$$
t^*(R)
=
\underset{t}{\arg\min}
\;
F(R,t)
$$

Global translation is unobservable because the objective depends only on
relative translations.

One translation gauge is therefore removed before solving the system.

Let $A$ be the oriented graph incidence matrix and define:

$$
\Omega
=
\mathrm{diag}
\left(
\tau_1,\ldots,\tau_m
\right)
$$

After removing one redundant row of the incidence matrix:

$$
B
=
A_{\mathrm{red}}
\Omega^{1/2}
$$

The projection matrix is:

$$
\Pi
=
I
-
B^T
(BB^T)^{-1}
B
$$

The inverse in this expression is **not explicitly formed** in the
implementation. The corresponding linear system is solved numerically.

The reduced rotation-only matrix is:

$$
Q
=
L_{\rho}
+
T^T
\Omega^{1/2}
\Pi
\Omega^{1/2}
T
$$

and the reduced objective becomes:

$$
F_{\mathrm{red}}(R)
=
\mathrm{tr}
\left(
R Q R^T
\right)
$$

The implementation verifies:

$$
F(R,t^*(R))
=
F_{\mathrm{red}}(R)
$$

to numerical precision.

Translation elimination therefore does not discard the translation
measurements. Their optimal contribution remains encoded in $Q$.

---

# SymForce Residual

The same SE-Sync edge residual is implemented symbolically with SymForce.

For one edge:

$$
r^R_{ij}
=
\sqrt{\kappa_{ij}}
\left(
R_j-R_i\widetilde{R}_{ij}
\right)
$$

and:

$$
r^t_{ij}
=
\sqrt{\tau_{ij}}
\left(
t_j-t_i-R_i\widetilde{t}_{ij}
\right)
$$

For two-dimensional rotations:

$$
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

The complete generated edge residual is:

$$
r_{ij}
=
\begin{bmatrix}
\sqrt{\kappa_{ij}}
\mathrm{vec}
\left(
R_j-R_i\widetilde{R}_{ij}
\right)
\\
\sqrt{\tau_{ij}}
\left(
t_j-t_i-R_i\widetilde{t}_{ij}
\right)
\end{bmatrix}
$$

with:

$$
r_{ij}\in\mathbb{R}^6
$$

The local edge state is:

$$
x_{ij}
=
\begin{bmatrix}
\theta_i &
t_{ix} &
t_{iy} &
\theta_j &
t_{jx} &
t_{jy}
\end{bmatrix}^T
$$

and the local Jacobian is:

$$
J_{ij}
=
\frac{\partial r_{ij}}
{\partial x_{ij}}
$$

with:

$$
J_{ij}\in\mathbb{R}^{6\times6}
$$

The SymForce implementation intentionally reproduces the SE-Sync
matrix-difference rotational residual rather than replacing it with a
different Lie-group pose residual.

---

# Sparse Global Linearization

For a graph with $n$ poses and $m$ measurements, the local Jacobians are
scattered into:

$$
J_c
\in
\mathbb{R}^{6m\times n}
$$

for angular variables, and:

$$
J_f
\in
\mathbb{R}^{6m\times2n}
$$

for translation variables.

The complete Jacobian is:

$$
J
=
\begin{bmatrix}
J_c & J_f
\end{bmatrix}
$$

and:

$$
H
=
J^T J
$$

with block structure:

$$
H
=
\begin{bmatrix}
H_{cc} & H_{cf} \\
H_{fc} & H_{ff}
\end{bmatrix}
$$

The implementation also verifies the exact global-translation nullspace:

$$
J_f v_x = 0
$$

and:

$$
J_f v_y = 0
$$

---

# Gauss-Newton Optimization

At nonlinear iteration $k$:

$$
r(x_k+\Delta x)
\approx
r_k+J_k\Delta x
$$

The local least-squares problem is:

$$
\min_{\Delta x}
\left\|
r_k+J_k\Delta x
\right\|_2^2
$$

Define:

$$
g=J^T r
$$

and:

$$
H=J^T J
$$

The symmetric Gauss-Newton step then satisfies:

$$
H\Delta x=-g
$$

Because the objective is:

$$
F=r^T r
$$

its gradient is:

$$
\nabla F=2J^T r
$$

and the Gauss-Newton Hessian approximation is:

$$
H_{\mathrm{GN}}
\approx
2J^T J
$$

The factor of two cancels from the linear solve.

---

# Frozen Linear-System Validation

Before comparing complete nonlinear optimizers, SparseQR and LDLT were
compared on identical frozen graph linearizations.

## MIT

```text
J               = 4962 x 2424
QR rank          = 2422

reduced H        = 2422 x 2422
nnz(reduced H)   = 13896

SparseQR total   ≈ 2175.43 ms
LDLT total       ≈    0.824 ms

measured ratio   ≈ 2641x
```

The two steps give effectively identical residual-space predictions:

```text
||J(dx_QR - dx_LDLT)|| ≈ 1.14e-11

relative ||H dx + g|| ≈ 2.46e-15
```

## CSAIL

```text
J               = 7032 x 3135
QR rank          = 3133

reduced H        = 3133 x 3133
nnz(reduced H)   = 19011

SparseQR total   ≈ 4604.54 ms
LDLT total       ≈    1.247 ms

measured ratio   ≈ 3693x
```

Residual-space prediction difference:

```text
≈ 5.17e-13
```

Normalized normal-equation residual:

```text
≈ 1.03e-15
```

---

# Official SE-Sync Validation

The official SE-Sync implementation was run independently on the same MIT
dataset.

It reports:

```text
measurements             = 827
poses                    = 808

chordal initial objective ≈ 88.1316
certified final objective ≈ 61.1541
```

The official solver reports convergence to a second-order critical point and
a global optimum, with the primal and dual SDP objectives agreeing to
numerical precision.

The rounded SE-Sync trajectory was exported and independently evaluated with
this repository.

Our implementation gives:

```text
rotation cost    = 41.0850352038581
translation cost = 20.0690808879390

total cost       = 61.1541160917971
```

The relative discrepancy from the official value is approximately:

```text
4.65e-16
```

This strongly validates the implemented:

- measurement convention,
- precision conversion,
- direct objective,
- SymForce residual,
- and graph-level evaluator.

---

# Local Solver Validation Near the Certified Solution

The official SE-Sync trajectory was used only as an external validation
initialization.

Starting at:

```text
61.154116091797107
```

the LDLT optimizer produces:

```text
iteration 0 : 61.154116091797107
iteration 1 : 61.154115526595191
iteration 2 : 61.154115524704849
iteration 3 : 61.154115524641696
iteration 4 : 61.154115524636858
```

Final comparison:

```text
QR final cost      = 61.1541155246367
LDLT final cost    = 61.1541155246369

iterations         = 4

QR solve total     ≈ 8936.441 ms
LDLT solve total   ≈    3.774 ms
```

This demonstrates that the local nonlinear implementation behaves correctly
in the neighborhood of the externally certified optimum.

---

# Initialization Sensitivity

The current MIT experiments expose an important distinction.

Starting from the raw MIT trajectory:

```text
initial objective
      |
      v
649214.84
      |
      v
Gauss-Newton
      |
      v
1298.03
```

Starting near the official SE-Sync solution:

```text
61.154116
     |
     v
Gauss-Newton
     |
     v
61.1541155
```

Therefore:

> The current MIT discrepancy is primarily an initialization / nonconvex
> optimization issue rather than an error in the SparseQR or LDLT linear
> backend.

---

# GTSAM Cross-Validation

A standalone GTSAM reference implementation was also tested.

Using Levenberg-Marquardt with a standard GTSAM `Pose2` factor graph:

```text
initial GTSAM error ≈ 3.548660355e9
final GTSAM error   ≈ 385.1195

iterations          = 29
wall time           ≈ 116.8 ms
```

When this trajectory is evaluated under the SE-Sync objective used in this
repository:

```text
SE-Sync cost ≈ 1360.8679
```

Conversely, the local SE-Sync trajectory near `1298` produces a substantially
different value when evaluated using the GTSAM factor graph.

The two implementations therefore optimize different numerical objectives.

GTSAM is useful as an external optimizer reference but is not used as the
ground-truth objective for this project.

---

# Current Research Direction

The validated solver stack now separates two problems.

## 1. Local optimization efficiency

Already investigated through:

- SparseQR,
- gauge-reduced normal equations,
- `SimplicialLDLT`,
- and future Schur-complement / structured elimination work.

## 2. Robustness to poor initialization

The current local Gauss-Newton solver can converge to a poor basin.

Chordal initialization will be retained as a practical baseline.

Additional directions under consideration include:

- Levenberg-Marquardt,
- trust-region optimization,
- multistart experiments,
- and certifiable global relaxation methods.

The key experimental target is:

```text
poor / arbitrary / unavailable pose initialization
                     |
                     v
             relative measurements
                     |
                     v
          optimization / relaxation
                     |
                     v
          trajectory near F = 61.1541
                     |
                     v
        optional global certification
```

---

# Future Schur-Complement Work

The local normal matrix can be partitioned as:

$$
H
=
\begin{bmatrix}
H_{cc} & H_{cf} \\
H_{fc} & H_{ff}
\end{bmatrix}
$$

After appropriate translation-gauge handling, the local Schur complement is:

$$
S
=
H_{cc}
-
H_{cf}
H_{ff}^{-1}
H_{fc}
$$

The inverse will not be formed explicitly. Instead, the translation block
will be factorized and applied through sparse linear solves.

The first validation target will be:

$$
\left\|
J
\left(
\Delta x_{\mathrm{full}}
-
\Delta x_{\mathrm{Schur}}
\right)
\right\|
\approx 0
$$

The local Gauss-Newton Schur complement should not be confused with the exact
SE-Sync reduced matrix $Q$.

They arise from different elimination problems.

---

# Repository Structure

```text
.
├── apps
│   ├── benchmark_optimizer_solvers.cpp
│   ├── export_solver_validation.cpp
│   ├── matrix_equivalence.cpp
│   ├── reference_pgo.cpp
│   └── simplified_equivalence.cpp
│
├── cmake
│   └── SymForceStage.cmake
│
├── codegen
│   └── python
│       └── generate_pgo_residuals.py
│
├── docs
│   └── figures
│       ├── mit_solver_validation.png
│       └── csail_solver_validation.png
│
├── generated
│   ├── edge_jacobian.h
│   └── edge_residual.h
│
├── include
│   └── cora_symforce
│
├── scripts
│   ├── csv_to_vertex_g2o.py
│   ├── evaluate_sesync_trajectory.py
│   └── plot_solver_validation.py
│
├── src
│
├── tests
│
├── tools
│   └── gtsam_reference
│
├── CMakeLists.txt
├── README.md
└── requirements-lock.txt
```

---

# Main Files

### `src/pgo_cost.cpp`

Independent handwritten implementation of the SE-Sync objective.

### `src/sesync_matrices.cpp`

Construction of the `B1`, `B2`, `B3`, and explicit quadratic formulations.

### `src/simplified_pgo.cpp`

Translation gauge removal, analytical translation elimination, and
construction of the reduced matrix $Q$.

### `codegen/python/generate_pgo_residuals.py`

Symbolic definition and differentiation of the exact SE-Sync edge residual.

### `src/symforce_pgo_cost.cpp`

Adapter between the C++ graph representation and generated SymForce kernels.

### `src/gauss_newton.cpp`

Nonlinear Gauss-Newton implementation and sparse linear-solver integration.

### `apps/benchmark_optimizer_solvers.cpp`

Repeated SparseQR versus LDLT nonlinear benchmarks.

### `apps/export_solver_validation.cpp`

Exports optimization histories, timing information, and trajectories.

### `scripts/plot_solver_validation.py`

Produces convergence, timing, trajectory, and trajectory-error plots.

### `tools/gtsam_reference`

External GTSAM optimizer and cost-evaluation utilities.

---

# Build

## Requirements

- C++17
- CMake
- Eigen3
- Python 3
- SymForce

The GTSAM tools are optional and are built separately.

---

## Generate SymForce Kernels

```bash
.venv/bin/python codegen/python/generate_pgo_residuals.py
```

## Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## Build

```bash
cmake --build build -j"$(nproc)"
```

---

# Tests

Run:

```bash
ctest \
  --test-dir build \
  --output-on-failure
```

Current regression status:

```text
11/11 tests passed
100% tests passed
0 tests failed
```

---

# Running MIT Solver Validation

```bash
./build/export_solver_validation \
  data/MIT.g2o \
  results/solver_validation/MIT \
  200
```

This experiment confirms that the local result near `1298` is not caused by
the previous 30-iteration limit.

---

# Out of Scope / Not Yet Implemented

The repository does not currently claim implementation of:

- an internal chordal initializer,
- the complete official SE-Sync solver,
- SDP relaxation,
- Burer-Monteiro optimization,
- Riemannian Staircase,
- global optimality certification,
- nonlinear Schur optimizer,
- SPARSER matrix-free computation,
- CORA range factors,
- range-state variables,
- or the complete CORA solver.

---

# Research Progression

```text
Direct SE-Sync objective
        |
        v
B1 / B2 / B3 equivalence
        |
        v
Explicit quadratic formulation
        |
        v
Exact translation elimination
        |
        v
Reduced Q objective
        |
        v
SymForce residual generation
        |
        v
Generated Jacobian validation
        |
        v
Sparse graph assembly
        |
        v
SparseQR Gauss-Newton
        |
        v
Gauge-reduced LDLT
        |
        v
Full nonlinear equivalence
        |
        v
MIT / CSAIL validation
        |
        v
External GTSAM comparison
        |
        v
Official SE-Sync validation
        |
        v
Initialization-sensitivity diagnosis
        |
        v
CURRENT STAGE
```

---

# References

This work builds on research in certifiable pose-graph optimization,
symbolic code generation, and robotic state estimation, including:

- **SE-Sync: A Certifiably Correct Algorithm for Synchronization over the
  Special Euclidean Group**
- **SymForce: Symbolic Computation and Code Generation for Robotics**
- **CORA: Certifiably Correct Range-Aided SLAM**

The original publications and upstream implementations should be consulted
for the complete algorithms and theoretical guarantees.

---

# Repository

[github.com/Prathmesh2931/cora-symforce-foundation](https://github.com/Prathmesh2931/cora-symforce-foundation)

---

## Current Takeaway

Three conclusions have now been established.

**1. The SE-Sync objective is reproduced correctly.**

The handwritten and SymForce implementations reproduce the official
SE-Sync objective at the certified MIT trajectory to approximately machine
precision.

**2. The symmetric LDLT backend reproduces the SparseQR nonlinear solution.**

The speed improvement observed in the current implementation does not produce
a different local trajectory.

**3. The remaining MIT issue is initialization sensitivity.**

The supplied MIT initialization converges to approximately:

```text
1298.03
```

while official SE-Sync obtains the certified solution near:

```text
61.1541
```

The next stage therefore focuses on recovering the desired solution under
poor or unavailable pose initialization while retaining the current
formulation and solver-validation stack as the numerical baseline.