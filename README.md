# CORA / SE-Sync SymForce Foundation

A C++ research implementation for reproducing and validating the SE-Sync pose-graph objective, generating exact residual/Jacobian kernels with SymForce, assembling sparse graph linearizations, and studying numerical optimization backends under controlled validation.

The current repository is focused on **SE(2) pose-graph optimization**. It does **not** yet include CORA range variables or a complete certifiable CORA solver. The purpose of this repository is to isolate and validate each mathematical and numerical layer before moving toward structured elimination, initialization-robust optimization, and certifiable/global formulations.

---

## 1. Research Question

The project started from a simple implementation question:

> Can the SE-Sync pose-graph objective be reproduced exactly, differentiated symbolically, assembled sparsely, and solved efficiently in C++ without changing the underlying mathematics?

That question has now been validated through several layers:

```text
SE-Sync objective
      |
      v
Handwritten residuals
      |
      v
B1 / B2 / B3 operators
      |
      v
Explicit quadratic matrix M
      |
      v
Exact translation elimination
      |
      v
Reduced rotation-only Q
      |
      v
SymForce symbolic residual / Jacobian
      |
      v
Generated C++ kernels
      |
      v
Sparse graph linearization
      |
      v
Gauss-Newton
      |
      +--> SparseQR reference backend
      |
      +--> gauge-reduced normal equations + SimplicialLDLT
      |
      v
Real MIT / CSAIL validation
      |
      v
External GTSAM + official SE-Sync cross-validation
```

The latest experiments exposed a second, more important question:

> How should a solver recover the correct solution when the supplied pose initialization lies in a poor nonlinear basin?

On the MIT SE(2) graph, the current local Gauss-Newton implementation converges reliably to a local solution near `1298.03` from the supplied vertices, while the official SE-Sync implementation certifies the global optimum near `61.1541`.

This distinction now separates two research threads:

1. **efficient local optimization**: sparse factorization, Schur/VarPro, matrix-free operators;
2. **initialization-robust / certifiable optimization**: chordal baselines, LM/trust-region alternatives, and reduced-\(Q\) global-relaxation approaches.

---

## 2. Pose-Graph Objective

For an edge \((i,j)\),

```math
F(R,t)
=
\sum_{(i,j)}
\kappa_{ij}
\left\|
R_j-R_i\tilde R_{ij}
\right\|_F^2
+
\tau_{ij}
\left\|
t_j-t_i-R_i\tilde t_{ij}
\right\|_2^2 .
```

where:

- \(R_i,R_j\in SO(2)\) are absolute rotations,
- \(t_i,t_j\in\mathbb{R}^2\) are absolute translations,
- \(\tilde R_{ij},\tilde t_{ij}\) are relative-pose measurements,
- \(\kappa_{ij}\) is rotational precision,
- \(\tau_{ij}\) is translational precision.

For `EDGE_SE2` information entries

```text
I11 I12 I13 I22 I23 I33
```

the loader follows the SE-Sync isotropic precision model:

```math
\tau
=
\frac{2}
{\operatorname{tr}
\left(
\operatorname{TranInfo}^{-1}
\right)},
\qquad
\operatorname{TranInfo}
=
\begin{bmatrix}
I_{11} & I_{12}\\
I_{12} & I_{22}
\end{bmatrix},
```

and

```math
\kappa = I_{33}.
```

The translation/rotation cross-information entries are not used by this SE-Sync objective.

---

## 3. Equivalent SE-Sync Formulations

### 3.1 Direct residuals

```math
e^R_{ij}
=
R_j-R_i\tilde R_{ij},
```

```math
e^t_{ij}
=
t_j-t_i-R_i\tilde t_{ij}.
```

The edge cost is

```math
F_{ij}
=
\kappa_{ij}\|e^R_{ij}\|_F^2
+
\tau_{ij}\|e^t_{ij}\|_2^2.
```

The handwritten implementation in `src/pgo_cost.cpp` remains an independent numerical reference.

### 3.2 \(B_1/B_2/B_3\) residual operators

The same objective is rewritten as

```math
e_t=B_1t+B_2r,
\qquad
e_R=B_3r,
```

with

```math
F
=
\|B_1t+B_2r\|_2^2
+
\|B_3r\|_2^2.
```

The operators encode

```math
B_1t
=
\sqrt{\tau_{ij}}(t_j-t_i),
```

```math
B_2r
=
-\sqrt{\tau_{ij}}R_i\tilde t_{ij},
```

and

```math
B_3r
=
\sqrt{\kappa_{ij}}
\operatorname{vec}
\left(
R_j-R_i\tilde R_{ij}
\right).
```

### 3.3 Explicit quadratic data matrix

With the state arranged as translations followed by rotation blocks,

```math
Y=
\left[
t_0\;t_1\;\cdots\;t_{n-1}
\mid
R_0\;R_1\;\cdots\;R_{n-1}
\right],
```

the objective is represented exactly as

```math
F(Y)=\operatorname{tr}(YMY^\top),
```

with block structure

```math
M=
\begin{bmatrix}
L_\tau & V\\
V^\top & L(G^\rho)+\Sigma
\end{bmatrix}.
```

No relaxation is introduced here. The direct residual, operator, and quadratic forms are numerically equivalent.

---

## 4. Exact Translation Elimination

For fixed rotations, translations form a linear least-squares subproblem:

```math
t^\star(R)
=
\arg\min_t F(R,t).
```

Because global translation is unobservable, one redundant translation degree of freedom is removed before factorization.

Using the reduced weighted incidence matrix

```math
B=A_{\mathrm{red}}\Omega^{1/2},
```

the projection operator is

```math
\Pi
=
I-B^\top(BB^\top)^{-1}B.
```

The inverse is **not** formed explicitly in code; the corresponding linear system is solved numerically.

The exact reduced rotation-only matrix is

```math
Q
=
L(G^\rho)
+
T^\top
\Omega^{1/2}
\Pi
\Omega^{1/2}
T,
```

giving

```math
F_{\mathrm{red}}(R)
=
\operatorname{tr}(RQR^\top).
```

The controlled three-pose validation gives:

```text
Direct PGO cost                  0.449279443492
Explicit cost at t*(R)           0.232564837850
Reduced Q cost                   0.232564837850
Explicit vs reduced difference   0
```

The implementation also verifies:

```text
Q  == Q^T
Pi == Pi^T
Pi^2 == Pi
```

### Important distinction

The exact SE-Sync matrix \(Q\) is **not** the same object as the Gauss-Newton normal matrix \(H=J^\top J\).

- \(Q\): exact global rotation-only data matrix after analytical translation elimination;
- \(H\): local tangent-space quadratic approximation around the current nonlinear estimate.

---

## 5. SymForce Residual and Generated Jacobian

For one edge,

```math
r_{ij}
=
\begin{bmatrix}
\sqrt{\kappa_{ij}}
\operatorname{vec}
\left(
R_j-R_i\tilde R_{ij}
\right)
\\
\sqrt{\tau_{ij}}
\left(
t_j-t_i-R_i\tilde t_{ij}
\right)
\end{bmatrix}
\in\mathbb{R}^{6}.
```

The local state ordering is

```math
x_{ij}
=
[
\theta_i,\,
t_{ix},\,
t_{iy},\,
\theta_j,\,
t_{jx},\,
t_{jy}
]^\top .
```

Therefore

```math
J_{ij}
=
\frac{\partial r_{ij}}
{\partial x_{ij}}
\in\mathbb{R}^{6\times6}.
```

The residual itself is manually defined from the SE-Sync equations. SymForce is used for symbolic differentiation, simplification, common-subexpression elimination, and C++ code generation.

The generated residual intentionally uses the SE-Sync matrix-difference rotational residual rather than a generic Lie-group `Log` / `local_coordinates` error.

### Edge-level numerical validation

```text
Residual equivalence error        2.002685474353e-16
Cost equivalence error            5.551115123126e-17
Jacobian finite-difference error  1.654302250742e-10
```

Additional differential checks:

```text
2 J^T r  vs finite-difference cost gradient   4.787481522328e-11
J v      vs finite-difference residual dir.   9.510081611097e-11
```

These checks independently verify both the generated residual and the derivative used by optimization.

---

## 6. Sparse Graph Linearization

Each generated \(6\times6\) edge Jacobian only depends on the two connected poses.

For \(n\) poses and \(m\) edges, the graph linearization scatters local derivatives into

```math
J_c\in\mathbb{R}^{6m\times n},
\qquad
J_f\in\mathbb{R}^{6m\times2n},
```

where:

- \(J_c\): one rotation-angle column per pose;
- \(J_f\): two translation columns per pose.

The Gauss-Newton blocks are

```math
H_{cc}=J_c^\top J_c,
```

```math
H_{cf}=J_c^\top J_f,
```

```math
H_{ff}=J_f^\top J_f.
```

The implementation explicitly validates the two global translation null directions:

```math
J_fv_x=0,
\qquad
J_fv_y=0.
```

### MIT graph structure

```text
poses                            808
edges                            827
Jc shape                         4962 x 808
Jf shape                         4962 x 1616
nnz(Jc)                          8262
nnz(Jf)                          3308
nnz(Hcc)                         2462
nnz(Hcf)                         3260
nnz(Hff)                         4924
||Jf vx||                        0
||Jf vy||                        0
```

---

## 7. Gauss-Newton Optimizer

At the current estimate \(x_k\),

```math
r(x_k+\Delta x)
\approx
r_k+J_k\Delta x.
```

The local problem is

```math
\min_{\Delta x}
\|r_k+J_k\Delta x\|_2^2.
```

For

```math
F=r^\top r,
```

the gradient is

```math
\nabla F=2J^\top r.
```

The implementation uses backtracking line search and records per-iteration:

- objective cost,
- gradient norm,
- step norm,
- accepted step length,
- sparse linearization time,
- linear-solve time.

---

## 8. Linear Solver Backends

### 8.1 SparseQR reference

The reference backend solves the rectangular least-squares system directly:

```math
J\Delta x\approx-r.
```

Eigen `SparseQR` is useful as a rank-revealing numerical reference.

For the MIT graph:

```text
state columns      2424
SparseQR rank      2422
rank deficiency    2
```

For CSAIL:

```text
state columns      3135
SparseQR rank      3133
rank deficiency    2
```

The observed deficiency matches the two exact global translation gauge directions.

### 8.2 Gauge-reduced symmetric normal equations

After fixing the first pose's translation increments,

```math
\Delta t_{0x}=0,
\qquad
\Delta t_{0y}=0,
```

the reduced system is

```math
H=J_r^\top J_r,
\qquad
g=J_r^\top r,
```

```math
H\Delta x_r=-g.
```

The current implementation solves this symmetric sparse system using Eigen `SimplicialLDLT`.

This backend is validated against SparseQR on the **same frozen residual and Jacobian**, rather than by comparing unrelated solver runs.

### Frozen-linearization validation

MIT:

```text
J shape                          4962 x 2424
nnz(J)                           11570
SparseQR rank                    2422
Reduced H shape                  2422 x 2422
nnz(H)                           13896
H symmetry max error             0
SparseQR total                   2175.43 ms
Symmetric path total             0.824 ms
QR / symmetric ratio             2641x
||J(dx_QR - dx_LDLT)||           1.14e-11
relative ||H dx + g||            2.46e-15
```

CSAIL:

```text
J shape                          7032 x 3135
nnz(J)                           16408
SparseQR rank                    3133
Reduced H shape                  3133 x 3133
nnz(H)                           19011
H symmetry max error             0
SparseQR total                   4604.54 ms
Symmetric path total             1.247 ms
QR / symmetric ratio             3693x
||J(dx_QR - dx_LDLT)||           5.17e-13
relative ||H dx + g||            1.03e-15
```

These results show that both methods produce effectively the same **local residual-space correction** on the tested linearizations.

### Numerical caveat

Normal equations trade speed for conditioning:

```math
\kappa(J^\top J)\approx\kappa(J)^2.
```

The LDLT path is therefore treated as a measured backend result, not as a universally superior replacement for QR.

---

## 9. Full Nonlinear QR vs LDLT

Both backends use the same:

- objective,
- generated Jacobians,
- initialization,
- line search,
- stopping rules,
- nonlinear update convention.

Only the linear algebra backend changes.

### CSAIL

Three-run median:

```text
Initial cost                      496.89616014905
Final cost                        31.703715883625
Iterations                        4

SparseQR solve total              18632.565 ms
LDLT solve total                  5.381 ms

SparseQR wall time                18635.963 ms
LDLT wall time                    8.605 ms

Relative final-cost gap           2.24e-15
QR / LDLT solve-time ratio        3462.6x
QR / LDLT wall-time ratio         2165.6x
```

### MIT

A 30-iteration comparison produced the same nonlinear cost trajectory and final objective for both backends, while the linear solver dominated the QR runtime.

A later diagnostic run increased the maximum iteration limit to 200. Both QR and LDLT stopped after **69 iterations** at

```text
QR final cost                     1298.03279183363
LDLT final cost                   1298.03279183363

QR solve total                    147450.190477 ms
LDLT solve total                  61.898108 ms

QR / LDLT solve-time ratio        ~2382x
```

This experiment established that the `~1298` MIT result was **not caused by the earlier 30-iteration cap**.

### Scope of the timing claim

These numbers compare two backends inside this implementation:

```text
Eigen SparseQR on J
vs
gauge-reduced J^T J + Eigen SimplicialLDLT
```

They are **not** a claim of outperforming GTSAM, g2o, or the official SE-Sync solver.

---

## 10. Solver / Trajectory Agreement

The validation exporter stores:

```text
qr_iterations.csv
ldlt_iterations.csv
qr_trajectory_trace.csv
ldlt_trajectory_trace.csv
qr_final.csv
ldlt_final.csv
metadata.txt
```

The comparison uses rigid SE(2) alignment with rotation + translation only; no scale is fitted.

Representative results:

### CSAIL

```text
relative final-cost gap           2.24e-15
aligned ATE RMSE                  2.76e-14 m
max position error                4.58e-14 m
heading RMSE                      4.74e-16 rad
```

### MIT

```text
relative final-cost gap           ~1e-15
aligned position disagreement     ~1e-12 m
heading disagreement              ~1e-14 rad
```

The QR and LDLT nonlinear trajectories are therefore numerically indistinguishable for the tested graphs.

---

## 11. External Validation: GTSAM

GTSAM was added as an external diagnostic reference to distinguish:

- objective mismatch,
- initialization sensitivity,
- local solver behavior.

### GTSAM Gauss-Newton

On the raw MIT initialization, GTSAM Gauss-Newton encountered an indeterminate / poorly conditioned system with its default Cholesky path.

Using a QR backend allowed the solve to proceed, but pure GN increased the GTSAM objective and stopped after one iteration.

### GTSAM Levenberg-Marquardt

Using LM + QR:

```text
Initial GTSAM error               3.548660355e9
Final GTSAM error                 385.119492
Iterations                        29
Wall time                         ~117 ms
```

However, GTSAM's standard Pose2 objective is **not the same scalar objective** as the SE-Sync matrix-difference formulation used here.

Cross-evaluation makes this explicit:

```text
Trajectory                  GTSAM objective       SE-Sync objective
-------------------------------------------------------------------
GTSAM LM solution              385.12                 1360.87
Our local GN/LDLT solution   48248.86                 1298.03
```

This experiment is diagnostic only. It demonstrates that low cost under one formulation cannot be compared directly with low cost under the other.

---

## 12. Official SE-Sync Reference and Global Optimum

The official SE-Sync implementation was run on the **same MIT dataset**.

It reported:

```text
Loaded measurements                 827
Poses                               808
Initialization                      chordal
Chordal initial objective           88.1316
Final objective                     61.1541
Riemannian gradient norm            3.604e-03
SDP duality gap                     -8.53e-14
Rounded-pose suboptimality bound    -4.74e-12
Result                              global optimum certified
```

The rounded official trajectory was exported and evaluated independently inside this repository.

### Independent Python evaluator

```text
Rotation cost                       41.0850352038581
Translation cost                    20.0690808879390
Total                               61.1541160917972
```

### Handwritten C++ and SymForce evaluators

```text
Reference PGO cost                  61.1541160917971
SymForce PGO cost                   61.1541160917971
Absolute difference                 2.84e-14
Relative difference                 4.65e-16
```

This is a strong end-to-end validation that:

1. the dataset precision conversion matches the intended SE-Sync objective;
2. the handwritten objective matches official SE-Sync's recovered trajectory;
3. the SymForce-generated residual matches the handwritten evaluator;
4. the remaining `1298` result from raw MIT vertices is not an objective-definition bug.

---

## 13. Initialization Sensitivity: Main Current Finding

Starting from the raw MIT `VERTEX_SE2` values:

```text
Initial SE-Sync cost                649214.841883749
Our GN final cost                   1298.03279183363
Iterations                          69
```

Starting from the GTSAM-LM trajectory:

```text
Initial SE-Sync cost                1360.86786651290
Our GN final cost                   1298.04314018996
Iterations                          59
```

Starting from the official certified SE-Sync trajectory:

```text
Initial SE-Sync cost                61.1541160917971
QR final cost                       61.1541155246367
LDLT final cost                     61.1541155246369
Iterations                          4
```

The important observation is therefore:

```text
very different poor starts
        |
        v
local Gauss-Newton
        |
        v
~1298 basin
```

while

```text
certified SE-Sync trajectory
        |
        v
our exact same GN / QR / LDLT machinery
        |
        v
~61.1541
```

So the current local residual, Jacobian, sparse assembly, and LDLT backend are capable of operating correctly around the globally optimal trajectory.

The open problem is **how to reach the correct basin without relying on a supplied high-quality pose estimate**.

---

## 14. Current Baselines and Next Research Direction

### Baseline A: local sparse optimization

```text
dataset / supplied trajectory
        ↓
SymForce residual + Jacobian
        ↓
sparse graph assembly
        ↓
GN + line search
        ↓
SparseQR or LDLT
```

This path is implemented and validated.

### Baseline B: chordal initialization + local optimization

Official SE-Sync demonstrates that chordal initialization starts the MIT problem near

```text
88.1316
```

rather than

```text
649214.84.
```

A local chordal + GN/LDLT implementation is therefore a useful practical baseline.

### Research direction under evaluation

The next direction is intentionally being separated into two possibilities.

#### A. Initialization-robust local methods

Candidate experiments:

- Levenberg-Marquardt,
- trust-region methods,
- continuation / damping,
- multistart,
- random / corrupted initial-pose stress tests.

These methods may escape poor local basins, but do not by themselves provide a global-optimality certificate.

#### B. Certifiable reduced-\(Q\) optimization

Because the exact translation-eliminated matrix \(Q\) is already derived and validated, the second direction is to study a global/certifiable relaxation of

```math
F_{\mathrm{red}}(R)
=
\operatorname{tr}(RQR^\top).
```

Possible implementation sequence:

```text
small explicit SDP reference
        ↓
validate against official SE-Sync
        ↓
low-rank factorization
        ↓
scalable Riemannian / structured solver
        ↓
round rotations
        ↓
recover translations analytically
        ↓
compare feasible objective with SDP lower bound
```

A separate alternative reference under consideration is certified rotation averaging followed by analytical translation recovery.

No SDP relaxation, low-rank staircase, or global certification is claimed as implemented in this repository yet.

---

## 15. Schur Complement / Variable Projection

The local Gauss-Newton normal matrix is partitioned as

```math
H
=
\begin{bmatrix}
H_{cc} & H_{cf}\\
H_{fc} & H_{ff}
\end{bmatrix},
\qquad
g
=
\begin{bmatrix}
g_c\\
g_f
\end{bmatrix}.
```

After gauge handling, translations can be locally eliminated:

```math
S
=
H_{cc}
-
H_{cf}
H_{ff}^{-1}
H_{fc},
```

with reduced right-hand side

```math
b_S
=
-g_c
+
H_{cf}H_{ff}^{-1}g_f.
```

The implementation should never explicitly form \(H_{ff}^{-1}\); factorization solves are used instead.

This remains an important performance direction, but it solves a different problem from initialization robustness:

```text
Schur / VarPro
    -> reduce computation

global / certifiable method
    -> reduce dependence on local nonlinear basin
```

The two directions can later be combined in a structured solver.

---

## 16. Repository Structure

```text
.
├── apps/
│   ├── benchmark_g2o.cpp
│   ├── benchmark_linear_solvers.cpp
│   ├── benchmark_optimizer_solvers.cpp
│   ├── export_solver_validation.cpp
│   ├── export_trajectory.cpp
│   ├── matrix_equivalence.cpp
│   ├── optimize_g2o.cpp
│   ├── reference_pgo.cpp
│   └── simplified_equivalence.cpp
│
├── cmake/
│   └── SymForceStage.cmake
│
├── codegen/python/
│   └── generate_pgo_residuals.py
│
├── generated/
│   ├── edge_jacobian.h
│   └── edge_residual.h
│
├── include/cora_symforce/
│   ├── g2o_dataset.hpp
│   ├── gauss_newton.hpp
│   ├── graph_linearization.hpp
│   ├── measurement.hpp
│   ├── pgo_cost.hpp
│   ├── pose2.hpp
│   ├── sesync_matrices.hpp
│   ├── simplified_pgo.hpp
│   ├── symmetric_solver.hpp
│   ├── symforce_pgo_cost.hpp
│   ├── toy_problem.hpp
│   └── trajectory_io.hpp
│
├── scripts/
│   ├── plot_solver_validation.py
│   ├── evaluate_sesync_trajectory.py
│   └── csv_to_vertex_g2o.py
│
├── src/
│   ├── g2o_dataset.cpp
│   ├── gauss_newton.cpp
│   ├── graph_linearization.cpp
│   ├── pgo_cost.cpp
│   ├── sesync_matrices.cpp
│   ├── simplified_pgo.cpp
│   ├── symmetric_solver.cpp
│   ├── symforce_pgo_cost.cpp
│   ├── toy_problem.cpp
│   └── trajectory_io.cpp
│
├── tests/
│   ├── test_g2o_dataset.cpp
│   ├── test_gauss_newton.cpp
│   ├── test_graph_linearization.cpp
│   ├── test_matrix_equivalence.cpp
│   ├── test_reference_pgo.cpp
│   ├── test_simplified_equivalence.cpp
│   ├── test_symmetric_solver.cpp
│   ├── test_symforce_edge_equivalence.cpp
│   ├── test_symforce_linearization.cpp
│   └── test_symforce_pgo_cost.cpp
│
├── tools/gtsam_reference/
│   ├── gtsam_reference.cpp
│   └── gtsam_cost_eval.cpp
│
├── CMakeLists.txt
└── README.md
```

Generated result folders and datasets are intentionally not part of the tracked source tree.

---

## 17. Main Implementation Files

### `src/pgo_cost.cpp`

Independent handwritten implementation of the weighted SE-Sync pose-graph objective.

### `src/sesync_matrices.cpp`

Constructs the \(B_1,B_2,B_3\) residual operators and explicit quadratic formulation.

### `src/simplified_pgo.cpp`

Implements exact translation elimination, gauge handling, projection \(\Pi\), and reduced rotation-only \(Q\).

### `codegen/python/generate_pgo_residuals.py`

Defines the exact SE-Sync edge residual symbolically and generates C++ residual/Jacobian kernels.

### `src/symforce_pgo_cost.cpp`

C++ adapter between project pose/measurement types and generated SymForce kernels.

### `src/graph_linearization.cpp`

Scatters edge-local Jacobians into global sparse \(J_c,J_f\) and forms local Gauss-Newton blocks.

### `src/gauss_newton.cpp`

Nonlinear Gauss-Newton loop, line search, convergence tests, and selectable SparseQR / symmetric LDLT backend.

### `src/symmetric_solver.cpp`

Controlled frozen-linearization QR-vs-LDLT comparison with rank, timing, residual-space equivalence, and normal-equation checks.

### `src/g2o_dataset.cpp`

Loads `VERTEX_SE2` / `EDGE_SE2` datasets and converts g2o information entries to the SE-Sync precision model.

### `src/trajectory_io.cpp`

Handles dataset trajectory initialization, measurement propagation for edge-only graphs, and trajectory export.

### `apps/export_solver_validation.cpp`

Exports nonlinear iteration traces and final trajectories for numerical/visual solver comparison.

### `tools/gtsam_reference/`

External GTSAM diagnostic tools used to compare objective behavior and nonlinear convergence.

---

## 18. Build

### Requirements

Core project:

- C++17
- CMake
- Eigen3
- Python 3
- SymForce for symbolic code generation

Generate the C++ kernels:

```bash
.venv/bin/python \
  codegen/python/generate_pgo_residuals.py
```

Configure and build:

```bash
cmake \
  -S . \
  -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build \
  build \
  -j"$(nproc)"
```

Run the regression suite:

```bash
ctest \
  --test-dir build \
  --output-on-failure
```

Current core regression status:

```text
100% tests passed
0 tests failed
11/11 tests passed
```

---

## 19. Useful Experiments

### Dataset formulation / sparse structure

```bash
./build/benchmark_g2o \
  data/MIT.g2o
```

### Long-run MIT convergence diagnosis

```bash
mkdir -p results/solver_validation/MIT_200

./build/export_solver_validation \
  data/MIT.g2o \
  results/solver_validation/MIT_200 \
  200
```

### Plot QR / LDLT agreement

```bash
python3 scripts/plot_solver_validation.py \
  results/solver_validation/MIT_200 \
  --label MIT
```

### Evaluate an external trajectory under the SE-Sync objective

```bash
python3 scripts/evaluate_sesync_trajectory.py \
  data/MIT.g2o \
  trajectory_vertices.g2o
```

---

## 20. What Has Been Established

The repository currently supports the following claims:

```text
[validated]
Direct PGO objective
       ==
B1 / B2 / B3 formulation
       ==
explicit quadratic M

[validated]
F(R, t*(R))
       ==
tr(R Q R^T)

[validated]
handwritten residual
       ==
SymForce-generated residual

[validated]
generated Jacobian
       ~= finite differences

[validated]
local SparseQR correction
       ~= gauge-reduced LDLT correction

[validated]
QR and LDLT nonlinear trajectories
       ~= numerical precision

[validated]
official SE-Sync certified MIT trajectory
       ==
our handwritten SE-Sync objective
       ==
our SymForce-generated objective
       ~= 61.1541161

[observed]
raw MIT vertices + local GN
       -> ~1298 local basin

[not yet implemented]
initialization-independent global/certifiable solver
```

---

## 21. Current Research Status

The project has moved beyond formulation reproduction.

The current evidence says:

1. **Objective correctness is established.**
2. **Generated residual/Jacobian correctness is established.**
3. **Sparse graph assembly is established.**
4. **QR-vs-LDLT equivalence is established.**
5. **The symmetric sparse backend removes a major measured linear-solve bottleneck on the tested SE(2) graphs.**
6. **Official SE-Sync cross-validation confirms the globally optimal MIT objective at approximately 61.1541.**
7. **The remaining scientific problem is not the local linear solve; it is nonlinear initialization / global recovery.**

The next implementation direction is being chosen around that final point, while chordal + local GN/LDLT remains the practical baseline and explicit Schur/VarPro remains the structured-computation path.

---

## 22. Scope / Non-Claims

This repository currently does **not** claim:

- a complete SE-Sync reimplementation;
- implementation of SE-Sync's Riemannian Staircase;
- an internal SDP global-optimality certificate;
- CORA range-variable optimization;
- SE(3) dataset support;
- superiority over GTSAM, g2o, SE-Sync, or other mature SLAM libraries;
- universal QR-to-LDLT speedups;
- robustness to arbitrary poor initialization.

The official SE-Sync implementation is currently used as the external certified reference for global-optimum validation.

---

## 23. Immediate Next Milestones

```text
1. Keep chordal + GN/LDLT as practical baseline
2. Stress-test poor / random / missing pose initialization
3. Choose between:
      a) robust local globalization
         (LM / trust-region / multistart)
      b) certifiable reduced-Q optimization
         (explicit SDP reference -> scalable low-rank method)
4. Validate against official SE-Sync
5. Implement explicit local Schur complement / VarPro
6. Study sparse or matrix-free reduced operators
7. Extend to SE(3) only after the 2D path is understood
```

The central target is no longer merely to make the local solve faster. It is to understand how solver structure, initialization, and global optimality interact while preserving an independently validated SE-Sync objective.
