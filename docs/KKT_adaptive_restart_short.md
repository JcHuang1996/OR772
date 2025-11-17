# KKT-Based Adaptive Restart (Short Version)

All notation (LP, variables, primal–dual iterates, averaging, outer/inner indices) follows the PDLP summary and the original PDLP paper; we only replace the restart metric with a KKT-based one and keep the same outer–inner restart structure.

## KKT Error

For a given outer iteration \(n\) with primal weight \(\omega_n > 0\) and \(z = (x, y, \varepsilon)\):

- Primal residual
\[
r_p(x) :=
\begin{bmatrix}
Ax - b \\
(h - Gx)_+
\end{bmatrix}.
\]

- Dual residual
\[
r_d(y,\varepsilon) := c - K^\top y - \varepsilon.
\]

- Duality gap term
\[
\mathrm{gap}(x,y,\varepsilon)
:= q^\top y + l^\top \varepsilon^+ - u^\top \varepsilon^- - c^\top x,
\quad
\varepsilon^+_i = \max\{0,\varepsilon_i\},\;
\varepsilon^-_i = \min\{0,\varepsilon_i\}.
\]

- KKT error (for outer iteration \(n\), weight \(\omega_n\)):
\[
\mathrm{KKT}_n(z)
=
\sqrt{
\omega_n^2 \,\| r_p(x) \|_2^2
+
\omega_n^{-2} \,\| r_d(y,\varepsilon) \|_2^2
+
\mathrm{gap}(x,y,\varepsilon)^2
}.
\]

## Restart Candidate

At inner step \(t+1\) of outer iteration \(n\), let
- \(z_{n,t+1}\) be the current PDHG iterate,
- \(\bar z_{n,t+1}\) be the averaged iterate.

The restart candidate is the one with smaller KKT error:
\[
z_{n,t+1}^c =
\begin{cases}
z_{n,t+1}, & \mathrm{KKT}_n(z_{n,t+1}) < \mathrm{KKT}_n(\bar z_{n,t+1}),\\[4pt]
\bar z_{n,t+1}, & \text{otherwise}.
\end{cases}
\]

## Restart Criteria

Let
- \(z_{n,0}\) be the starting point of outer iteration \(n\),
- \(t\) be the current inner index in outer iteration \(n\),
- \(k\) be the total number of inner iterations so far,
- \(t_c\) be the inner index at which the **previous** restart candidate was recorded.

Define
\[
K_0 := \mathrm{KKT}_n(z_{n,0}),\quad
K_c := \mathrm{KKT}_n(z_{n,t+1}^c),\quad
K_{\mathrm{last}} := \mathrm{KKT}_n(z_{n,t_c}).
\]

Given parameters
\(\vartheta_{\mathrm{sufficient}}, \vartheta_{\mathrm{necessary}}, \vartheta_{\mathrm{artificial}} \in (0,1)\)
(e.g. \(0.2, 0.8, 0.36\)), **restart outer iteration \(n\)** at inner step \(t+1\) if **any** of:

1. **Sufficient decay**
   \[
   K_c \le \vartheta_{\mathrm{sufficient}} K_0.
   \]

2. **Necessary decay + no local progress**
   \[
   K_c \le \vartheta_{\mathrm{necessary}} K_0
   \quad\text{and}\quad
   K_c > K_{\mathrm{last}}.
   \]

3. **Long inner loop**
   \[
   t \ge \vartheta_{\mathrm{artificial}} \, k.
   \]

Otherwise, continue the current outer iteration without restart.
