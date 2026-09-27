# CORA / SE-Sync SymForce Foundation

A small C++ reference implementation for understanding and validating the optimization formulations behind SE-Sync before extending the work toward CORA and SymForce-generated C++.

The current focus is not a full SE-Sync or CORA implementation. The goal is to build each formulation step from the original pose-graph objective and verify that the different representations are numerically equivalent.

## Current Scope

The repository currently implements a small 2D pose-graph problem with relative rotation and translation measurements.

For an edge `(i, j)`, the reference PGO objective is

\[
F(R,t) =
\sum_{(i,j)}
\kappa_{ij}
\|R_j - R_i \tilde{R}_{ij}\|_F^2
+
\tau_{ij}
\|t_j - t_i - R_i \tilde{t}_{ij}\|_2^2.
\]

The implementation then follows the same objective through three representations.

### 1. Direct PGO residuals

The rotation and translation residuals are evaluated directly in C++.

\[
e^R_{ij} = R_j - R_i\tilde{R}_{ij}
\]

\[
e^t_{ij} = t_j - t_i - R_i\tilde{t}_{ij}
\]

A small three-pose graph is used as a controlled numerical reference.

### 2. SE-Sync matrix representation

The direct residuals are rewritten using the SE-Sync-style matrices

\[
e_t = B_1 t + B_2 r
\]

and

\[
e_R = B_3 r,
\]

where `r` contains the vectorized rotation blocks.

The explicit quadratic data matrix `M` is also constructed so that

\[
F(Y) = \operatorname{tr}(Y M Y^\top),
\]

with

\[
Y = [t_0 \; \cdots \; t_n \mid R_0 \; \cdots \; R_n].
\]

The implementation verifies numerically that

\[
F_{\text{direct}}
=
F_{B_1,B_2,B_3}
=
F_M.
\]

### 3. Simplified SE-Sync formulation

For fixed rotations, translations are treated as a linear least-squares subproblem

\[
t^\star(R)
=
\arg\min_t F(R,t).
\]

After removing the global translation gauge, the projection

\[
\Pi
=
I -
B^\top(BB^\top)^{-1}B
\]

is used to construct the reduced rotation-only matrix

\[
Q =
L(G^\rho)
+
T^\top
\Omega^{1/2}
\Pi
\Omega^{1/2}
T.
\]

The reduced objective is

\[
F_{\text{simplified}}(R)
=
\operatorname{tr}(R Q R^\top).
\]

The current numerical experiment verifies

\[
F(R,t^\star(R))
=
F_{\text{simplified}}(R).
\]

Translation measurements are therefore not discarded in the simplified formulation; their optimal contribution is retained in `Q`.

## Numerical Validation

For the current perturbed three-pose test problem:

```text
Direct PGO cost:                  0.449279443492
Explicit cost at t*(R):           0.232564837850
Simplified Q cost:                0.232564837850
Explicit vs simplified diff:      0.000000000000