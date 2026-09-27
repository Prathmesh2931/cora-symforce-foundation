# CORA / SE-Sync SymForce Foundation

A small C++ reference implementation for studying and validating the pose-graph optimization formulations used in SE-Sync, with the longer-term goal of extending the same validated formulation toward CORA and SymForce-generated C++.

The current repository is intentionally focused on the formulation rather than a full SE-Sync or CORA solver.

## Current Scope

The current implementation uses a small 2D pose graph with relative rotation and translation measurements.

For an edge `(i, j)`, the reference pose-graph objective is

$$
F(R,t) =
\sum_{(i,j)}
\kappa_{ij}
\left\|R_j - R_i \tilde{R}_{ij}\right\|_F^2
+
\tau_{ij}
\left\|t_j - t_i - R_i \tilde{t}_{ij}\right\|_2^2.
$$

The same objective is then evaluated through multiple equivalent representations.

---

## 1. Direct PGO Formulation

The relative pose residuals are implemented directly in C++.

Rotation residual:

$$
e^R_{ij}
=
R_j - R_i \tilde{R}_{ij}
$$

Translation residual:

$$
e^t_{ij}
=
t_j - t_i - R_i \tilde{t}_{ij}
$$

A small three-pose graph is used as a controlled numerical reference.

At the state used to generate the measurements, both residuals are zero.

---

## 2. SE-Sync Matrix Formulation

The direct residuals are rewritten using the SE-Sync-style matrices `B1`, `B2`, and `B3`.

For the translation residuals,

$$
e_t = B_1 t + B_2 r
$$

and for the rotation residuals,

$$
e_R = B_3 r,
$$

where `r` contains the vectorized rotation blocks.

The implementation verifies that the matrix residuals match the direct residuals numerically.

The explicit state is arranged as

$$
Y =
\left[
t_0 \; t_1 \; \cdots \; t_{n-1}
\mid
R_0 \; R_1 \; \cdots \; R_{n-1}
\right].
$$

The explicit quadratic data matrix `M` is constructed such that

$$
F(Y)
=
\operatorname{tr}
\left(
Y M Y^\top
\right).
$$

The current implementation verifies

$$
F_{\text{direct}}
=
F_{B_1,B_2,B_3}
=
F_M.
$$

---

## 3. Simplified SE-Sync Formulation

For fixed rotations, translations form a linear least-squares subproblem:

$$
t^\star(R)
=
\arg\min_t F(R,t).
$$

Because the objective only depends on relative translations, global translation is unobservable. A translation gauge is therefore removed before solving the reduced system.

Using the reduced weighted incidence matrix `B`, the projection is

$$
\Pi
=
I -
B^\top
(BB^\top)^{-1}
B.
$$

The reduced rotation-only matrix is then

$$
Q
=
L(G^\rho)
+
T^\top
\Omega^{1/2}
\Pi
\Omega^{1/2}
T.
$$

The simplified objective becomes

$$
F_{\text{simplified}}(R)
=
\operatorname{tr}
\left(
R Q R^\top
\right).
$$

The implementation verifies

$$
F(R,t^\star(R))
=
F_{\text{simplified}}(R).
$$

This means the simplified formulation eliminates the translation variables, but does **not** discard translation measurement information. Their optimal contribution remains encoded in `Q`.

---

## Numerical Validation

For the current perturbed three-pose example:

```text
Direct PGO cost:                  0.449279443492
Explicit cost at t*(R):           0.232564837850
Simplified Q cost:                0.232564837850
Explicit vs simplified diff:      0.000000000000
```

The implementation also checks:

```text
Direct translation residual == B1 * t + B2 * r
Direct rotation residual    == B3 * r

Direct PGO cost == explicit M cost

Q  == Q^T
Pi == Pi^T
Pi^2 == Pi
```

All current regression tests pass.

---

## Repository Structure

```text
.
├── apps
│   ├── matrix_equivalence.cpp
│   ├── reference_pgo.cpp
│   └── simplified_equivalence.cpp
│
├── codegen
│   └── python
│
├── generated
│
├── include
│   └── cora_symforce
│       ├── measurement.hpp
│       ├── pgo_cost.hpp
│       ├── pose2.hpp
│       ├── sesync_matrices.hpp
│       ├── simplified_pgo.hpp
│       └── toy_problem.hpp
│
├── src
│   ├── pgo_cost.cpp
│   ├── sesync_matrices.cpp
│   ├── simplified_pgo.cpp
│   └── toy_problem.cpp
│
└── tests
    ├── test_matrix_equivalence.cpp
    ├── test_reference_pgo.cpp
    └── test_simplified_equivalence.cpp
```

### Main implementation files

- `src/pgo_cost.cpp`  
  Direct weighted pose-graph objective.

- `src/sesync_matrices.cpp`  
  Construction of `B1`, `B2`, `B3`, and the explicit data matrix `M`.

- `src/simplified_pgo.cpp`  
  Translation elimination, projection `Pi`, and construction of the reduced matrix `Q`.

The programs in `apps/` are small numerical experiments used to validate the equivalence between these formulations.

---

## Build

Requirements:

- C++17
- CMake
- Eigen3

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
```

Run the experiments:

```bash
./build/reference_pgo
./build/matrix_equivalence
./build/simplified_equivalence
```

Run all tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## Current Progress

The formulation path currently validated is

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
Simplified matrix Q
```

No SDP, Burer-Monteiro, Riemannian Staircase, or CORA range factor has been implemented yet.

---

## Next Step

The next step is to reproduce the validated pose-graph residual using SymForce as a symbolic and code-generation layer.

The intended workflow is

```text
Validated handwritten C++ residual
              |
              v
      SymForce symbolic model
              |
              v
       Generated C++
              |
              v
 Numerical equivalence checks
              |
              v
 Jacobian / runtime experiments
```

CORA range residuals and the later certifiable optimization layers will be introduced only after the base pose-graph formulation remains numerically consistent.