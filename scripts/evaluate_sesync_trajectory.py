#!/usr/bin/env python3

import math
import sys

import numpy as np


def R(theta):
    c = math.cos(theta)
    s = math.sin(theta)

    return np.array(
        [[c, -s],
         [s,  c]],
        dtype=float,
    )


def load_vertices(path):
    poses = {}

    with open(path) as f:
        for line in f:
            parts = line.split()

            if not parts:
                continue

            if parts[0] != "VERTEX_SE2":
                continue

            idx = int(parts[1])

            poses[idx] = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )

    return poses


def load_edges(path):
    edges = []

    with open(path) as f:
        for line in f:
            parts = line.split()

            if not parts:
                continue

            if parts[0] != "EDGE_SE2":
                continue

            i = int(parts[1])
            j = int(parts[2])

            dx = float(parts[3])
            dy = float(parts[4])
            dtheta = float(parts[5])

            I11 = float(parts[6])
            I12 = float(parts[7])
            I22 = float(parts[9])
            I33 = float(parts[11])

            Tinfo = np.array(
                [
                    [I11, I12],
                    [I12, I22],
                ],
                dtype=float,
            )

            Tcov = np.linalg.inv(Tinfo)

            tau = 2.0 / np.trace(Tcov)
            kappa = I33

            edges.append(
                (
                    i,
                    j,
                    dx,
                    dy,
                    dtheta,
                    tau,
                    kappa,
                )
            )

    return edges


def evaluate(poses, edges):
    rot_cost = 0.0
    trans_cost = 0.0

    for (
        i,
        j,
        dx,
        dy,
        dtheta,
        tau,
        kappa,
    ) in edges:

        xi, yi, thi = poses[i]
        xj, yj, thj = poses[j]

        Ri = R(thi)
        Rj = R(thj)

        Rij = R(dtheta)

        ti = np.array([xi, yi])
        tj = np.array([xj, yj])

        tij = np.array([dx, dy])

        eR = Rj - Ri @ Rij

        et = (
            tj -
            ti -
            Ri @ tij
        )

        rot_cost += (
            kappa *
            float(np.sum(eR * eR))
        )

        trans_cost += (
            tau *
            float(et @ et)
        )

    return (
        rot_cost,
        trans_cost,
        rot_cost + trans_cost,
    )


if len(sys.argv) != 3:
    print(
        "usage: evaluate_sesync_trajectory.py "
        "<measurement_graph.g2o> "
        "<trajectory_vertices.g2o>"
    )
    raise SystemExit(1)


poses = load_vertices(sys.argv[2])
edges = load_edges(sys.argv[1])

rotation, translation, total = evaluate(
    poses,
    edges,
)

print("Poses                  :", len(poses))
print("Edges                  :", len(edges))
print("SE-Sync rotation cost  :", f"{rotation:.15g}")
print("SE-Sync translation cost:", f"{translation:.15g}")
print("SE-Sync total cost     :", f"{total:.15g}")
