from dataclasses import dataclass, field
import numpy as np

@dataclass
class WeightedStatistics:
    """Uncentered weighted cross-products for one set of rows.

    Uncentered on purpose: centred moments cannot be added, so folds could not be
    combined or subtracted. Centring happens in `centred`, once the subset is known.
    """

    features: int
    rows: int = 0
    mass: float = 0.0
    sum_x: np.ndarray = field(default=None)
    sum_xx: np.ndarray = field(default=None)
    sum_y: float = 0.0
    sum_xy: np.ndarray = field(default=None)
    sum_yy: float = 0.0

    def __post_init__(self) -> None:
        if self.sum_x is None:
            self.sum_x = np.zeros(self.features, dtype=np.float64)
        if self.sum_xx is None:
            self.sum_xx = np.zeros((self.features, self.features), dtype=np.float64)
        if self.sum_xy is None:
            self.sum_xy = np.zeros(self.features, dtype=np.float64)

    def add(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Accumulate a block of rows. X is promoted to float64 for the products."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if X.shape[1] != self.features:
            raise ValueError(f"Expected {self.features} features, got {X.shape[1]}")
        weighted = X * w[:, None]
        self.rows += len(X)
        self.mass += float(w.sum())
        self.sum_x += weighted.sum(axis=0)
        self.sum_xx += weighted.T @ X
        self.sum_y += float(w @ y)
        self.sum_xy += weighted.T @ y
        self.sum_yy += float(w @ (y ** 2))

    def __add__(self, other: "WeightedStatistics") -> "WeightedStatistics":
        return self._combine(other, +1.0)

    def __sub__(self, other: "WeightedStatistics") -> "WeightedStatistics":
        return self._combine(other, -1.0)

    def _combine(self, other: "WeightedStatistics", sign: float) -> "WeightedStatistics":
        if other.features != self.features:
            raise ValueError("Cannot combine statistics with different feature counts")
        return WeightedStatistics(
            features=self.features, rows=self.rows + int(sign) * other.rows,
            mass=self.mass + sign * other.mass,
            sum_x=self.sum_x + sign * other.sum_x,
            sum_xx=self.sum_xx + sign * other.sum_xx,
            sum_y=self.sum_y + sign * other.sum_y,
            sum_xy=self.sum_xy + sign * other.sum_xy,
            sum_yy=self.sum_yy + sign * other.sum_yy)

    def centred(self):
        """Return (C, s, syy, x_mean, y_mean) with weights normalised to total mass 1."""
        if self.mass <= 0:
            raise ValueError("Statistics carry no mass")
        x_mean = self.sum_x / self.mass
        y_mean = self.sum_y / self.mass
        C = self.sum_xx / self.mass - np.outer(x_mean, x_mean)
        C = (C + C.T) * 0.5          # kill accumulation drift, as stage 2 does
        s = self.sum_xy / self.mass - x_mean * y_mean
        syy = self.sum_yy / self.mass - y_mean * y_mean
        return C, s, float(syy), x_mean, float(y_mean)

    def weighted_sse(self, intercept: float, coefficients: np.ndarray) -> float:
        """Weighted sum of squared errors for y ~ intercept + x . coefficients.

        Expanding the quadratic lets a fold be scored from its statistics alone, so
        held-out error needs no second pass over the rows.
        """
        c = np.asarray(coefficients, dtype=np.float64)
        return float(self.sum_yy
                     - 2.0 * intercept * self.sum_y
                     - 2.0 * (c @ self.sum_xy)
                     + intercept ** 2 * self.mass
                     + 2.0 * intercept * (c @ self.sum_x)
                     + c @ self.sum_xx @ c)
