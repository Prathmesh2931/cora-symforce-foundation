#!/usr/bin/env python3
"""Visual validation for SparseQR vs symmetric LDLT Gauss-Newton runs.

SparseQR is treated as the numerical reference baseline for this comparison.
This script does NOT provide a global-optimality certificate and does NOT treat
SparseQR as ground truth.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_iteration_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key != "iteration":
                row[key] = float(row[key])
        row["iteration"] = int(row["iteration"])
    return rows


def read_trace(path):
    by_iteration = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["iteration"])
            by_iteration.setdefault(k, []).append(
                (int(row["pose_id"]), float(row["x"]), float(row["y"]), float(row["theta"]))
            )
    result = {}
    for k, rows in by_iteration.items():
        rows.sort(key=lambda x: x[0])
        result[k] = {
            "xy": np.array([[r[1], r[2]] for r in rows], dtype=float),
            "theta": np.array([r[3] for r in rows], dtype=float),
        }
    return result


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def align_se2(target_xy, target_theta, reference_xy, reference_theta):
    if target_xy.shape != reference_xy.shape:
        raise ValueError("Trajectory shapes do not match.")

    xt = target_xy - target_xy.mean(axis=0)
    xr = reference_xy - reference_xy.mean(axis=0)
    u, _, vt = np.linalg.svd(xt.T @ xr)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T

    aligned_xy = xt @ rot.T + reference_xy.mean(axis=0)
    phi = math.atan2(rot[1, 0], rot[0, 0])
    aligned_theta = wrap_angle(target_theta + phi)
    theta_error = wrap_angle(aligned_theta - reference_theta)
    return aligned_xy, aligned_theta, theta_error, phi


def trajectory_metrics(qr, ldlt):
    common = sorted(set(qr) & set(ldlt))
    metrics = []
    for k in common:
        ref = qr[k]
        est = ldlt[k]
        aligned_xy, _, theta_error, _ = align_se2(
            est["xy"], est["theta"], ref["xy"], ref["theta"]
        )
        pos_error = np.linalg.norm(aligned_xy - ref["xy"], axis=1)
        metrics.append(
            {
                "iteration": k,
                "ate_rmse": float(np.sqrt(np.mean(pos_error ** 2))),
                "max_pos": float(np.max(pos_error)),
                "heading_rmse": float(np.sqrt(np.mean(theta_error ** 2))),
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--label", default="Pose graph")
    args = parser.parse_args()

    d = args.trace_dir
    qr_it = read_iteration_csv(d / "qr_iterations.csv")
    ldlt_it = read_iteration_csv(d / "ldlt_iterations.csv")
    qr_trace = read_trace(d / "qr_trajectory_trace.csv")
    ldlt_trace = read_trace(d / "ldlt_trajectory_trace.csv")

    metrics = trajectory_metrics(qr_trace, ldlt_trace)
    final_k = metrics[-1]["iteration"]
    ref = qr_trace[final_k]
    est = ldlt_trace[final_k]
    aligned_xy, _, final_theta_error, phi = align_se2(
        est["xy"], est["theta"], ref["xy"], ref["theta"]
    )
    final_pos_error = np.linalg.norm(aligned_xy - ref["xy"], axis=1)

    qr_cost = {r["iteration"]: r["cost"] for r in qr_it}
    ldlt_cost = {r["iteration"]: r["cost"] for r in ldlt_it}
    common_cost = sorted(set(qr_cost) & set(ldlt_cost))

    fig = plt.figure(figsize=(13, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot([r["iteration"] for r in qr_it], [r["cost"] for r in qr_it], marker="o", label="SparseQR")
    ax1.plot([r["iteration"] for r in ldlt_it], [r["cost"] for r in ldlt_it], marker="x", label="Symmetric LDLT")
    ax1.set_yscale("log")
    ax1.set_xlabel("Gauss-Newton iteration")
    ax1.set_ylabel("Objective cost")
    ax1.set_title("Convergence")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    qr_nonzero = qr_it[1:]
    ldlt_nonzero = ldlt_it[1:]
    ax2.plot([r["iteration"] for r in qr_nonzero], [r["cumulative_solve_ms"] for r in qr_nonzero], marker="o", label="SparseQR")
    ax2.plot([r["iteration"] for r in ldlt_nonzero], [r["cumulative_solve_ms"] for r in ldlt_nonzero], marker="x", label="Symmetric LDLT")
    ax2.set_yscale("log")
    ax2.set_xlabel("Gauss-Newton iteration")
    ax2.set_ylabel("Cumulative linear solve time [ms]")
    ax2.set_title("Solver time accumulation")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(ref["xy"][:, 0], ref["xy"][:, 1], label="SparseQR reference")
    ax3.plot(aligned_xy[:, 0], aligned_xy[:, 1], linestyle="--", label="LDLT aligned")
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_title("Final trajectory agreement")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot([m["iteration"] for m in metrics], [m["ate_rmse"] for m in metrics], marker="o", label="Position ATE RMSE [m]")
    ax4.plot([m["iteration"] for m in metrics], [m["heading_rmse"] for m in metrics], marker="x", label="Heading RMSE [rad]")
    ax4.set_yscale("log")
    ax4.set_xlabel("Gauss-Newton iteration")
    ax4.set_ylabel("QR-reference error")
    ax4.set_title("LDLT vs QR trajectory error")
    ax4.grid(True)
    ax4.legend()

    fig.suptitle(f"{args.label}: SparseQR vs symmetric LDLT validation")
    fig.tight_layout()
    out_png = d / "solver_validation_summary.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    final_cost_gap = abs(qr_cost[common_cost[-1]] - ldlt_cost[common_cost[-1]]) / max(
        1.0, abs(qr_cost[common_cost[-1]]), abs(ldlt_cost[common_cost[-1]])
    )
    final_ate = float(np.sqrt(np.mean(final_pos_error ** 2)))
    final_max = float(np.max(final_pos_error))
    final_heading = float(np.sqrt(np.mean(final_theta_error ** 2)))

    out_txt = d / "solver_validation_metrics.txt"
    with open(out_txt, "w") as f:
        f.write(f"label={args.label}\n")
        f.write("reference=SparseQR numerical baseline (not ground truth/certification)\n")
        f.write(f"common_final_iteration={final_k}\n")
        f.write(f"relative_final_cost_gap={final_cost_gap:.17e}\n")
        f.write(f"final_aligned_ATE_RMSE_m={final_ate:.17e}\n")
        f.write(f"final_aligned_max_position_error_m={final_max:.17e}\n")
        f.write(f"final_aligned_heading_RMSE_rad={final_heading:.17e}\n")
        f.write(f"alignment_rotation_rad={phi:.17e}\n")

    print(f"Saved {out_png}")
    print(f"Saved {out_txt}")
    print(f"Relative final-cost gap      : {final_cost_gap:.6e}")
    print(f"Final aligned ATE RMSE [m]   : {final_ate:.6e}")
    print(f"Final max position error [m] : {final_max:.6e}")
    print(f"Final heading RMSE [rad]     : {final_heading:.6e}")
    print("Reference: SparseQR numerical baseline, not a global certificate.")


if __name__ == "__main__":
    main()
