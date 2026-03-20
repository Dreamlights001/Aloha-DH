"""Lightweight 3D math helpers with no third-party dependencies."""

from __future__ import annotations

import math
from typing import List, Sequence

Vector = List[float]
Matrix = List[List[float]]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vector_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def vector_sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def vector_scale(a: Sequence[float], s: float) -> Vector:
    return [x * s for x in a]


def vector_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(n: int) -> Matrix:
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def transpose(m: Matrix) -> Matrix:
    rows = len(m)
    cols = len(m[0]) if rows > 0 else 0
    t = zeros(cols, rows)
    for r in range(rows):
        for c in range(cols):
            t[c][r] = m[r][c]
    return t


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    rows = len(a)
    inner = len(a[0]) if rows else 0
    cols = len(b[0]) if b else 0
    out = zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[r][k] * b[k][c]
            out[r][c] = total
    return out


def mat_vec_mul(a: Matrix, v: Sequence[float]) -> Vector:
    return [sum(a[r][c] * v[c] for c in range(len(v))) for r in range(len(a))]


def solve_linear_system(a: Matrix, b: Sequence[float]) -> Vector:
    """Solve Ax=b with Gaussian elimination + partial pivoting."""
    n = len(a)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix in solve_linear_system")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-15:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def homogeneous_translation(x: float, y: float, z: float) -> Matrix:
    t = identity(4)
    t[0][3] = x
    t[1][3] = y
    t[2][3] = z
    return t
