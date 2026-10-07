from __future__ import annotations

import numpy as np


def _validate_pair(
    y_true: np.ndarray,
    y_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_base = np.asarray(y_base, dtype=np.float64)

    if y_true.ndim != 2 or y_base.ndim != 2:
        raise ValueError(
            "Expected two-dimensional condition-by-gene matrices"
        )

    if y_true.shape != y_base.shape:
        raise ValueError(
            f"Shape mismatch: true={y_true.shape}, "
            f"base={y_base.shape}"
        )

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains non-finite values")

    if not np.isfinite(y_base).all():
        raise ValueError("y_base contains non-finite values")

    return y_true, y_base


class GlobalMeanResidual:
    """Add one training-set residual mean to every prediction."""

    def __init__(self) -> None:
        self.mean_residual: float | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_base: np.ndarray,
    ) -> "GlobalMeanResidual":
        y_true, y_base = _validate_pair(y_true, y_base)
        residual = y_true - y_base
        self.mean_residual = float(np.mean(residual))
        return self

    def predict(self, y_base: np.ndarray) -> np.ndarray:
        if self.mean_residual is None:
            raise RuntimeError("Model has not been fitted")

        y_base = np.asarray(y_base, dtype=np.float64)

        if y_base.ndim != 2:
            raise ValueError("Expected a two-dimensional matrix")

        return y_base + self.mean_residual


class GeneMeanResidual:
    """Add one training-set mean residual for each gene."""

    def __init__(self) -> None:
        self.mean_residual: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_base: np.ndarray,
    ) -> "GeneMeanResidual":
        y_true, y_base = _validate_pair(y_true, y_base)
        residual = y_true - y_base
        self.mean_residual = np.mean(
            residual,
            axis=0,
            keepdims=True,
        )
        return self

    def predict(self, y_base: np.ndarray) -> np.ndarray:
        if self.mean_residual is None:
            raise RuntimeError("Model has not been fitted")

        y_base = np.asarray(y_base, dtype=np.float64)

        if y_base.ndim != 2:
            raise ValueError("Expected a two-dimensional matrix")

        if y_base.shape[1] != self.mean_residual.shape[1]:
            raise ValueError(
                "Gene dimension differs from the fitted model"
            )

        return y_base + self.mean_residual


class GeneWiseAffine:
    """Fit one regularized affine calibration per gene.

    For gene g:

        y_true[:, g] ≈ slope[g] * y_base[:, g] + intercept[g]

    The intercept is not regularized. The ridge coefficient only
    regularizes the slope around zero.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.alpha = float(alpha)
        self.slopes: np.ndarray | None = None
        self.intercepts: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_base: np.ndarray,
    ) -> "GeneWiseAffine":
        y_true, y_base = _validate_pair(y_true, y_base)

        n_genes = y_true.shape[1]
        slopes = np.zeros(n_genes, dtype=np.float64)
        intercepts = np.zeros(n_genes, dtype=np.float64)

        for gene_idx in range(n_genes):
            x = y_base[:, gene_idx]
            y = y_true[:, gene_idx]

            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))

            centered_x = x - x_mean
            centered_y = y - y_mean

            numerator = float(
                np.dot(centered_x, centered_y)
            )
            denominator = float(
                np.dot(centered_x, centered_x)
                + self.alpha
            )

            if denominator <= 1e-12:
                slope = 0.0
            else:
                slope = numerator / denominator

            intercept = y_mean - slope * x_mean

            slopes[gene_idx] = slope
            intercepts[gene_idx] = intercept

        self.slopes = slopes
        self.intercepts = intercepts
        return self

    def predict(self, y_base: np.ndarray) -> np.ndarray:
        if self.slopes is None or self.intercepts is None:
            raise RuntimeError("Model has not been fitted")

        y_base = np.asarray(y_base, dtype=np.float64)

        if y_base.ndim != 2:
            raise ValueError("Expected a two-dimensional matrix")

        if y_base.shape[1] != len(self.slopes):
            raise ValueError(
                "Gene dimension differs from the fitted model"
            )

        return (
            y_base * self.slopes[None, :]
            + self.intercepts[None, :]
        )
