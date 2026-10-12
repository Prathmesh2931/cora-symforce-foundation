# CORA / SE-Sync SymForce Foundation

A small C++ reference implementation for understanding, reproducing, and numerically validating the pose-graph optimization formulations used in SE-Sync, together with a SymForce symbolic code-generation layer.

The longer-term goal is to use this validated foundation to study structured variable elimination, sparse Schur-complement methods, SPARSER-style matrix-free computation, and eventually broader CORA formulations.

The repository is intentionally focused on formulation, symbolic generation, and numerical equivalence rather than implementing the complete SE-Sync or CORA solver.

---

## Current Scope

The current implementation uses a small 2D pose graph with relative rotation and translation measurements.

For an edge `(i, j)`, the pose-graph objective is

```math
F(R,t)
=
\sum_{(i,j)}
\kappa_{ij}
\left\|R_j - R_i \tilde{R}_{ij}\right\|_F^2
+
\tau_{ij}
\left\|t_j - t_i - R_i \tilde{t}_{ij}\right\|_2^2
```

where

- `R_i`, `R_j` are absolute rotations,
- `t_i`, `t_j` are absolute translations,
- `\tilde{R}_{ij}` is the measured relative rotation,
- `\tilde{t}_{ij}` is the measured relative translation expressed in frame `i`,
- `\kappa_{ij}` is the rotational precision,
- `\tau_{ij}` is the translational precision.

The same objective is currently evaluated through several equivalent representations:

```text
Direct residuals
        |
        v
B1 / B2 / B3 residual operators
        |
        v
Explicit quadratic matrix M
        |
        v
Analytical translation elimination
        |
        v
Reduced rotation-only Q
        |
        v
SymForce symbolic residual
        |
        v
Generated C++ residual / Jacobian
        |
        v
Edge-level and graph-level equivalence tests
```

---

## 1. Direct PGO Formulation

The relative pose residuals are implemented directly in C++.

Rotation residual:

```math
e^R_{ij}
=
R_j - R_i \tilde{R}_{ij}
```

Translation residual:

```math
e^t_{ij}
=
t_j - t_i - R_i \tilde{t}_{ij}
```

The weighted edge cost is

```math
F_{ij}
=
\kappa_{ij}
\left\|e^R_{ij}\right\|_F^2
+
\tau_{ij}
\left\|e^t_{ij}\right\|_2^2
```

A small three-pose graph is used as a controlled numerical reference.

At the state used to generate the measurements, both residuals are zero. Perturbed states are then used for the equivalence tests so that the compared costs are non-zero.

---

## 2. SE-Sync Matrix Formulation

The direct residuals are rewritten using SE-Sync-style residual operators `B1`, `B2`, and `B3`.

For the translation residuals,

```math
e_t
=
B_1 t + B_2 r
```

and for the rotation residuals,

```math
e_R
=
B_3 r
```

where `r` contains the column-major vectorization of the rotation blocks.

The operators represent

```math
B_1 t
=
\sqrt{\tau_{ij}}
(t_j - t_i)
```

```math
B_2 r
=
-\sqrt{\tau_{ij}}
R_i \tilde{t}_{ij}
```

```math
B_3 r
=
\sqrt{\kappa_{ij}}
\operatorname{vec}
\left(
R_j - R_i \tilde{R}_{ij}
\right)
```

Therefore,

```math
F
=
\left\|B_1t + B_2r\right\|_2^2
+
\left\|B_3r\right\|_2^2
```

The implementation verifies numerically that the matrix residuals match the direct residuals.

The explicit state is arranged as

```math
Y
=
\left[
t_0 \; t_1 \; \cdots \; t_{n-1}
\mid
R_0 \; R_1 \; \cdots \; R_{n-1}
\right]
```

The explicit quadratic data matrix `M` is constructed such that

```math
F(Y)
=
\mathrm{tr}
\left(
YMY^\top
\right)
```

with block structure

```math
M
=
\begin{bmatrix}
L_\tau & V \\
V^\top & L(G^\rho) + \Sigma
\end{bmatrix}
```

where

- `L_\tau` is the translational graph Laplacian,
- `V` contains translation-rotation coupling terms,
- `\Sigma` contains the contribution from the squared translation-measurement term,
- `L(G^\rho)` is the rotational connection Laplacian.

The implementation verifies

```math
F_{\mathrm{direct}}
=
F_{B_1,B_2,B_3}
=
F_M
```

No relaxation is introduced at this stage. These are equivalent representations of the same PGO objective.

---

## 3. Simplified SE-Sync Formulation

For fixed rotations, translations form a linear least-squares subproblem:

```math
t^\star(R)
=
\arg\min_t F(R,t)
```

The objective depends only on relative translations, so global translation is unobservable.

A translation gauge is therefore removed before solving the reduced system.

Let `A` be the oriented incidence matrix and

```math
\Omega
=
\operatorname{diag}
(\tau_1,\ldots,\tau_m)
```

be the translational precision matrix.

After removing one redundant row of the incidence matrix,

```math
B
=
A_{\mathrm{red}}
\Omega^{1/2}
```

The projection matrix is

```math
\Pi
=
I
-
B^\top
(BB^\top)^{-1}
B
```

In the implementation, the inverse is not formed explicitly. The linear system involving `BB^T` is solved using a numerical factorization.

The reduced rotation-only matrix is

```math
Q
=
L(G^\rho)
+
T^\top
\Omega^{1/2}
\Pi
\Omega^{1/2}
T
```

The first term contains the rotational measurement contribution.

The second term retains the effect of the translation measurements after the translation variables have been eliminated.

The simplified objective is

```math
F_{\mathrm{simplified}}(R)
=
\mathrm{tr}
\left(
RQR^\top
\right)
```

The implementation verifies

```math
F(R,t^\star(R))
=
F_{\mathrm{simplified}}(R)
```

Therefore, the simplified formulation eliminates the translation variables but does not discard translation measurement information. Their optimal contribution remains encoded in `Q`.

---

## 4. SymForce Symbolic Residuals

The validated handwritten SE-Sync residual is reproduced symbolically using SymForce.

The purpose of this stage is not to introduce a different pose error or a different optimization objective.

The exact same weighted SE-Sync residual is used.

For one edge `(i,j)`:

```math
r^R_{ij}
=
\sqrt{\kappa_{ij}}
\left(
R_j - R_i\tilde{R}_{ij}
\right)
```

```math
r^t_{ij}
=
\sqrt{\tau_{ij}}
\left(
t_j - t_i - R_i\tilde{t}_{ij}
\right)
```

For the current 2D implementation,

```math
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
```

The rotation residual is vectorized column-major:

```math
\operatorname{vec}(r^R_{ij})
=
\begin{bmatrix}
r^R_{00} \\
r^R_{10} \\
r^R_{01} \\
r^R_{11}
\end{bmatrix}
```

The complete generated edge residual is therefore

```math
r_{ij}
=
\begin{bmatrix}
\sqrt{\kappa_{ij}}
\operatorname{vec}
\left(
R_j - R_i\tilde{R}_{ij}
\right)
\\
\sqrt{\tau_{ij}}
\left(
t_j - t_i - R_i\tilde{t}_{ij}
\right)
\end{bmatrix}
\in
\mathbb{R}^6
```

The edge state ordering used for symbolic differentiation is

```math
x_{ij}
=
\begin{bmatrix}
\theta_i &
t_{ix} &
t_{iy} &
\theta_j &
t_{jx} &
t_{jy}
\end{bmatrix}^{\top}
```

Therefore, the generated edge Jacobian has dimension

```math
J_{ij}
=
\frac{\partial r_{ij}}
{\partial x_{ij}}
\in
\mathbb{R}^{6\times6}
```

The SymForce implementation intentionally uses the SE-Sync matrix-difference rotational residual.

It does **not** replace it with a generic Lie-group logarithmic or `local_coordinates` pose residual.

---

## 5. Symbolic Jacobian Structure

Define

```math
J_2
=
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
```

For SO(2),

```math
\frac{\partial R(\theta)}
{\partial\theta}
=
R(\theta)J_2
```

Therefore, the translation residual has derivatives

```math
\frac{\partial r^t_{ij}}
{\partial t_i}
=
-\sqrt{\tau_{ij}}I
```

```math
\frac{\partial r^t_{ij}}
{\partial t_j}
=
+\sqrt{\tau_{ij}}I
```

and

```math
\frac{\partial r^t_{ij}}
{\partial\theta_i}
=
-\sqrt{\tau_{ij}}
R_iJ_2\tilde{t}_{ij}
```

The rotational residual derivatives are

```math
\frac{\partial r^R_{ij}}
{\partial\theta_i}
=
-\sqrt{\kappa_{ij}}
R_iJ_2\tilde{R}_{ij}
```

```math
\frac{\partial r^R_{ij}}
{\partial\theta_j}
=
+\sqrt{\kappa_{ij}}
R_jJ_2
```

The generated Jacobian is structurally sparse because each residual depends only on the variables associated with the two connected poses.

---

## 6. SymForce Code-Generation Pipeline

The symbolic implementation lives in

```text
codegen/python/generate_pgo_residuals.py
```

The current pipeline is

```text
Symbolic SE-Sync residual in Python
              |
              v
Symbolic differentiation
              |
              v
Common-subexpression elimination
              |
              v
Generated Eigen/C++ kernels
              |
              +--> generated/edge_residual.h
              |
              +--> generated/edge_jacobian.h
```

The generated files are runtime kernels and should normally not be edited manually.

Repeated symbolic expressions are extracted into temporary variables such as

```text
_tmp0
_tmp1
_tmp2
...
```

during code generation.

The generated kernels are wrapped by

```text
src/symforce_pgo_cost.cpp
```

which adapts the existing project data types to the generated SymForce functions.

For the current SO(2) implementation, the wrapper recovers the angle associated with an existing rotation matrix using

```math
\theta
=
\operatorname{atan2}
(R_{10},R_{00})
```

and then evaluates the generated residual or Jacobian.

---

## 7. SymForce Numerical Validation

Two independent levels of validation are currently implemented.

### 7.1 Edge-Level Residual and Jacobian Validation

The handwritten edge residual is compared directly against the SymForce-generated residual.

The scalar edge cost is also compared using

```math
F_{ij}
=
\left\|r_{ij}\right\|_2^2
```

The generated Jacobian is independently checked using central finite differences:

```math
J_k^{\mathrm{FD}}
=
\frac{
r(x+\epsilon e_k)
-
r(x-\epsilon e_k)
}{
2\epsilon
}
```

with

```math
\epsilon
=
10^{-6}
```

Current numerical results:

```text
Residual equivalence error:        2.002685474353e-16
Cost equivalence error:            5.551115123126e-17
Jacobian finite-difference error:  1.654302250742e-10
```

These differences are consistent with floating-point numerical precision.

### 7.2 Full Graph Validation

A separate three-pose, three-edge graph is used to verify that the generated edge kernels also reproduce the complete PGO objective.

The graph contains

```text
0 -> 1
1 -> 2
0 -> 2
```

Current result:

```text
Reference full PGO cost:  0.243919071312716
SymForce full PGO cost:   0.243919071312716
```

The per-edge rotational, translational, and total costs agree to approximately machine precision.

---

## 8. Numerical Validation Summary

For the original perturbed three-pose SE-Sync formulation:

```text
Direct PGO cost:                  0.449279443492
Explicit cost at t*(R):           0.232564837850
Simplified Q cost:                0.232564837850
Explicit vs simplified diff:      0.000000000000
```

The implementation checks

```text
Direct translation residual == B1 * t + B2 * r
Direct rotation residual    == B3 * r

Direct PGO cost == B1/B2/B3 cost
Direct PGO cost == explicit M cost

Explicit cost at t*(R) == simplified Q cost

Q  == Q^T
Pi == Pi^T
Pi^2 == Pi
```

The SymForce stage additionally checks

```text
Handwritten edge residual == generated SymForce residual
Handwritten edge cost     == generated SymForce edge cost

Generated Jacobian ~= central finite-difference Jacobian

Handwritten full graph cost == SymForce full graph cost
```

Current CTest status:

```text
5/5 tests passed
```

---

## Repository Structure

```text
.
├── apps
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
├── generated
│   ├── edge_jacobian.h
│   └── edge_residual.h
│
├── include
│   └── cora_symforce
│       ├── measurement.hpp
│       ├── pgo_cost.hpp
│       ├── pose2.hpp
│       ├── sesync_matrices.hpp
│       ├── simplified_pgo.hpp
│       ├── symforce_pgo_cost.hpp
│       └── toy_problem.hpp
│
├── src
│   ├── pgo_cost.cpp
│   ├── sesync_matrices.cpp
│   ├── simplified_pgo.cpp
│   ├── symforce_pgo_cost.cpp
│   └── toy_problem.cpp
│
├── tests
│   ├── test_matrix_equivalence.cpp
│   ├── test_reference_pgo.cpp
│   ├── test_simplified_equivalence.cpp
│   ├── test_symforce_edge_equivalence.cpp
│   └── test_symforce_pgo_cost.cpp
│
├── CMakeLists.txt
├── README.md
└── requirements-lock.txt
```

---

## Main Implementation Files

### `src/pgo_cost.cpp`

Direct implementation of the weighted pose-graph residuals and objective.

### `src/sesync_matrices.cpp`

Construction of `B1`, `B2`, `B3`, and the explicit quadratic data matrix `M`.

### `src/simplified_pgo.cpp`

Construction of the graph matrices, translation gauge removal, translation elimination, projection `Pi`, and reduced rotation-only matrix `Q`.

### `codegen/python/generate_pgo_residuals.py`

Defines the exact weighted SE-Sync residual symbolically, computes its Jacobian, and generates the C++ residual and Jacobian kernels.

### `generated/edge_residual.h`

Autogenerated C++ implementation of the symbolic edge residual.

### `generated/edge_jacobian.h`

Autogenerated C++ implementation of the symbolic edge Jacobian.

### `src/symforce_pgo_cost.cpp`

Adapter between the existing C++ pose/measurement structures and the generated SymForce kernels.

### `tests/test_symforce_edge_equivalence.cpp`

Checks generated residual equivalence, cost equivalence, and generated Jacobian accuracy against central finite differences.

### `tests/test_symforce_pgo_cost.cpp`

Checks full-graph equivalence between the handwritten PGO implementation and the SymForce-generated implementation.

---

## Build

### Requirements

- C++17
- CMake
- Eigen3
- Python 3
- SymForce for symbolic code generation

The Python environment used for code generation is isolated from the C++ runtime.

Install the Python dependencies in the project virtual environment as required by `requirements-lock.txt`.

Generate the SymForce C++ kernels:

```bash
.venv/bin/python codegen/python/generate_pgo_residuals.py
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
```

Run the formulation experiments:

```bash
./build/reference_pgo
./build/matrix_equivalence
./build/simplified_equivalence
```

Run the SymForce validation executables:

```bash
./build/test_symforce_edge_equivalence
./build/test_symforce_pgo_cost
```

Run the complete regression suite:

```bash
ctest --test-dir build --output-on-failure
```

Expected current result:

```text
100% tests passed, 0 tests failed out of 5
```

---

## Current Progress

The formulation and generated-code path currently implemented is

```text
Direct PGO residuals
        |
        v
B1 / B2 / B3 representation
        |
        v
Explicit quadratic matrix M
        |
        v
Translation elimination
        |
        v
Projector Pi
        |
        v
Reduced rotation-only Q
        |
        v
Exact SE-Sync residual in SymForce
        |
        v
Generated C++ residual
        |
        +------------------+
        |                  |
        v                  v
Generated Jacobian    C++ PGO wrapper
        |                  |
        v                  v
Finite-difference     Full-graph
validation            equivalence
        |                  |
        +--------+---------+
                 |
                 v
           Current stage
              complete
```

The repository currently establishes numerical equivalence between

```math
F_{\mathrm{direct}}
=
F_{B_1,B_2,B_3}
=
F_M
```

and, after translation elimination,

```math
F(R,t^\star(R))
=
F_Q(R)
```

while independently validating

```math
r_{\mathrm{handwritten}}
\approx
r_{\mathrm{SymForce}}
```

```math
J_{\mathrm{SymForce}}
\approx
J_{\mathrm{finite\ difference}}
```

and

```math
F_{\mathrm{handwritten\ graph}}
\approx
F_{\mathrm{SymForce\ graph}}
```

---

## Next Step: Schur Complement / Variable Projection

The next implementation stage is structured variable elimination using a block Schur complement.

The explicit quadratic formulation can be partitioned into translation and rotation blocks:

```math
M
=
\begin{bmatrix}
M_{tt} & M_{tr} \\
M_{rt} & M_{rr}
\end{bmatrix}
```

where translations are the variables to be eliminated.

Because global translation introduces a gauge freedom, the translation block must first be made full rank through the same gauge-removal principle already used in the simplified SE-Sync formulation.

After gauge removal, the translation solution has the form

```math
t^\star
=
-
M_{tt}^{-1}
M_{tr}r
```

and substitution produces the rotation-only Schur complement

```math
Q_{\mathrm{Schur}}
=
M_{rr}
-
M_{rt}
M_{tt}^{-1}
M_{tr}
```

The first validation goal will be

```math
Q_{\mathrm{Schur}}
\approx
Q
```

where `Q` is the already validated projector-based reduced SE-Sync matrix.

This will establish two equivalent views of the same analytical translation elimination:

```text
Projector formulation
        |
        v
Pi = I - B^T(BB^T)^(-1)B
        |
        v
Reduced Q

        ==

Block quadratic formulation
        |
        v
Translation Schur elimination
        |
        v
Reduced Q_Schur
```

Only after this equivalence is established will the implementation move toward sparse or matrix-free application of the reduced operator, following the later SPARSER-style research direction.

---

## Out of Scope / Not Yet Implemented

The repository does **not** currently claim implementation of:

- SPARSER matrix-free Schur application,
- Schur-based preconditioning,
- velocity or other non-pairwise residual extensions,
- CORA range residuals,
- SDP relaxation,
- Burer-Monteiro factorization,
- Riemannian Staircase,
- global optimality certification,
- a complete SE-Sync solver,
- a complete CORA solver.

The current objective is to build each formulation layer with independent numerical checks before progressing to the next optimization stage.

---

## Research Direction

The current research progression is

```text
Validated SE-Sync formulation
        |
        v
Validated SymForce residual / Jacobian generation
        |
        v
Schur-complement equivalence
        |
        v
Sparse / matrix-free reduced operator
        |
        v
Solver and preconditioning experiments
        |
        v
Non-pairwise / velocity residual investigation
        |
        v
Broader CORA integration
```

The handwritten SE-Sync implementation remains the numerical reference while SymForce is used as a symbolic differentiation and C++ code-generation layer.