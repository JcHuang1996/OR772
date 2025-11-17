```
import numpy as np

def _trust_region_linear(z, g, l, r, tol=1e-10, max_iter=60):
    """
    Solve:  minimize g^T z_hat
           s.t. ||z_hat - z||_2 <= r,   z_hat >= l  (componentwise)

    Uses a 1D search on λ with the closed-form
        z_hat(λ) = max(l, z - λ g),
    and finds λ such that ||z_hat(λ) - z||_2 = r (or until bounds saturate).
    """
    z = np.asarray(z, dtype=float)
    g = np.asarray(g, dtype=float)
    l = np.asarray(l, dtype=float)
    assert z.shape == g.shape == l.shape

    if r <= 0:
        return z, float(g @ z)

    def dist_at(lam):
        z_hat = np.maximum(z - lam * g, l)
        d = np.linalg.norm(z_hat - z)
        return d, z_hat

    # At λ = 0, we are at the center.
    lam_low = 0.0
    d_low, z_hat_low = dist_at(lam_low)
    if d_low >= r:
        # Already at or outside the ball (shouldn't really happen in normal use)
        return z_hat_low, float(g @ z_hat_low)

    # Grow λ until we reach / exceed the ball, or movement saturates.
    lam_high = 1.0
    d_high, z_hat_high = dist_at(lam_high)
    it = 0
    while d_high < r and it < 60:
        lam_high *= 2.0
        d_high, z_hat_high = dist_at(lam_high)
        it += 1

    if d_high < r:
        # All coordinates have hit their lower bounds before reaching radius r.
        # Then the optimum is just that saturated point.
        return z_hat_high, float(g @ z_hat_high)

    # Bisection on [lam_low, lam_high].
    z_hat_best = z_hat_low
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_low + lam_high)
        d_mid, z_hat_mid = dist_at(lam_mid)
        if d_mid > r:
            lam_high = lam_mid
        else:
            lam_low = lam_mid
            z_hat_best = z_hat_mid

    return z_hat_best, float(g @ z_hat_best)


def normalized_duality_gap_lp(A, b, c, x, y, z_ref=None, r=None, tol=1e-10):
    """
    Normalized duality gap for the LP:

        min  c^T x     s.t. A x = b, x >= 0
        max  b^T y     s.t. A^T y <= c

    at the current point z = (x, y), using the Euclidean norm.

    Parameters
    ----------
    A : (m, n) array_like
    b : (m,) array_like
    c : (n,) array_like
    x : (n,) array_like  (current primal point, not necessarily feasible)
    y : (m,) array_like  (current dual point, not necessarily feasible)
    z_ref : (n+m,) array_like, optional
        Reference point; if provided and r is None, we set r = ||z - z_ref||_2.
        This matches the PDLP choice μ_n(z, z_ref) = ρ_{||z-z_ref||}(z).
    r : float, optional
        Radius for ρ_r(z). If None, uses ||z - z_ref||_2 if z_ref is given,
        otherwise uses ||z||_2.
    tol : float, optional
        Numerical tolerance used inside the trust-region solver.

    Returns
    -------
    rho : float
        Normalized duality gap ρ_r(z).
    r_used : float
        The radius r actually used.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, n = A.shape
    assert x.shape == (n,)
    assert y.shape == (m,)

    # Form z and choose radius r.
    z = np.concatenate([x, y])
    if z_ref is not None:
        z_ref = np.asarray(z_ref, dtype=float).reshape(-1)
        assert z_ref.shape == z.shape
        r_val = np.linalg.norm(z - z_ref) if r is None else float(r)
    else:
        r_val = np.linalg.norm(z) if r is None else float(r)

    if r_val == 0.0:
        # Limit case: if radius is zero, ρ_0 is 0 at an optimal solution.
        # As a reasonable fallback, return the absolute primal-dual gap.
        return abs(float(c @ x - b @ y)), 0.0

    # Build the linear objective g for the trust-region LP:
    #   g_x = c - A^T y,   g_y = A x - b
    # and bounds l_x = 0, l_y = -inf (dual variables free).
    g_x = c - A.T @ y          # shape (n,)
    g_y = A @ x - b            # shape (m,)
    g = np.concatenate([g_x, g_y])

    l_x = np.zeros_like(x)
    l_y = np.full_like(y, -np.inf)
    l = np.concatenate([l_x, l_y])

    # Solve the trust-region subproblem to get T = min g^T z_hat
    z_hat_star, T = _trust_region_linear(z, g, l, r_val, tol=tol)

    # Constant term c^T x - b^T y comes from expanding L(x, ŷ) - L(x̂, y).
    C = float(c @ x - b @ y)

    rho = (C - T) / r_val
    return rho, r_val
```

