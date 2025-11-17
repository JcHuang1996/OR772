# PDLP (Primal–Dual Hybrid Gradient for Linear Programming): Main Algorithm and First Three Modules

This document states PDLP from the LP saddle-point form, then specifies the main loop and the first three modules (step size, adaptive restarts, primal weight). .

## Saddle-point formulation

- Solve the convex–concave saddle-point problem
  - $\min_{x\in X}\ \max_{y\in Y}\ L(x,y)$ with
    - $L(x,y) := c^\top x - y^\top K x + q^\top y$,
    - $X := \{x\in\mathbb{R}^n: l \le x \le u\}$,
    - $Y := \{y\in \mathbb{R}^{m_1+m_2}: y_{1:m_1}\ge 0\}$,
    - $K^\top := \begin{pmatrix} G^\top & A^\top \end{pmatrix}$ and $q^\top := \begin{pmatrix} h^\top & b^\top \end{pmatrix}$,
    - $Z := X \times Y$.
- Baseline PDHG iteration specialized to $L(x,y)$
  - $x_{k+1} = \operatorname{proj}_X\big(x_k - \tau\,(c - K^\top y_k)\big)$,
  - $y_{k+1} = \operatorname{proj}_Y\big(y_k + \sigma\,(q - K(2x_{k+1}-x_k))\big)$.
- Reparameterize step sizes by a step size $\eta>0$ and a primal weight $\omega>0$
  - $\tau = \eta/\omega$, $\ \sigma = \omega\eta$; PDHG converges for $\eta \le 1/\|K\|_2$.
  - Weighted norm $\|z\|_\omega := \sqrt{\,\omega\|x\|_2^2 + \|y\|_2^2/\omega\,}$ for $z=(x,y)$.

## PDLP: Main outer/inner loop (excluding presolve and scaling)

- Inputs and initialization
  - Input: initial point $z_{0,0}=(x_{0,0},y_{0,0})$.
  - Set outer-loop counter $n\leftarrow 0$; total-iteration counter $k\leftarrow 0$.
  - Initialize candidate step size $\hat\eta_{0,0}\leftarrow 1/\|K\|_\infty$.
  - Initialize primal weight $\omega_0 \leftarrow \mathrm{InitializePrimalWeight}(c,q)$.
- Outer loop over restart epochs ($n=0,1,2,\dots$)
  - Set inner counter $t\leftarrow 0$.
  - Inner loop (PDHG steps with dynamic step size)
    - One safeguarded PDHG step with the step-size heuristic
      - $(z_{n,t+1},\ \eta_{n,t+1},\ \hat\eta_{n,t+1}) \leftarrow \mathrm{AdaptiveStepOfPDHG}(z_{n,t},\ \omega_n,\ \hat\eta_{n,t},\ k)$.
    - Weighted running average of inner iterates
      - $\displaystyle \bar z_{n,t+1} \leftarrow \frac{\sum_{i=1}^{t+1} \eta_{n,i}\, z_{n,i}}{\sum_{i=1}^{t+1} \eta_{n,i}}$.
    - Select a restart candidate using the normalized duality gap (Module 2)
      - $z^{c}_{n,t+1} \leftarrow \mathrm{GetRestartCandidate}(z_{n,t+1},\ \bar z_{n,t+1},\ z_{n,0})$.
    - Update counters $t\leftarrow t+1$, $k\leftarrow k+1$.
    - Continue inner loop until the restart or termination criterion is met.
  - On restart
    - Set the next epoch start point $z_{n+1,0}\leftarrow z^{c}_{n,t}$ and $n\leftarrow n+1$.
    - Update primal weight $\omega_n\leftarrow \mathrm{PrimalWeightUpdate}(z_{n,0},\ z_{n-1,0},\ \omega_{n-1})$.
  - Stop when termination criteria (below) hold; output the current $z_{n,0}$.

## Module 1 — Adaptive step size: AdaptiveStepOfPDHG

Purpose: choose an accepted PDHG step and a next step-size candidate while enforcing the small-step bound.

Procedure (invoked at inner iteration $(n,t)$ with inputs $z_{n,t}=(x,y)$, $\omega_n$, $\hat\eta_{n,t}$, and global iteration $k$)

- Initialize local variables: set $\eta \leftarrow \hat\eta_{n,t}$.
- Repeat until acceptance
  - Primal update: $x' \leftarrow \operatorname{proj}_X\big(x - (\eta/\omega_n)\,(c - K^\top y)\big)$.
  - Dual update: $y' \leftarrow \operatorname{proj}_Y\big(y + (\eta\,\omega_n)\,(q - K(2x' - x))\big)$.
  - Safe bound on the step size
    - $\displaystyle \bar\eta \leftarrow \frac{\|(x'-x,\ y'-y)\|_{\omega_n}^2}{2\,(y'-y)^\top K\,(x'-x)}$.
  - Candidate for the next trial
    - $\displaystyle \eta' \leftarrow \min\Big((1-(k+1)^{-0.3})\,\bar\eta,\ \ (1+(k+1)^{-0.6})\,\eta\Big)$.
  - Accept or shrink
    - If $\eta \le \bar\eta$, accept and return $(x',y')$, the accepted step size $\eta$, and the next candidate $\eta'$.
    - Else set $\eta\leftarrow \eta'$ and repeat.

Rationale

- The step size must satisfy
  - $\displaystyle \eta \le \frac{\|z_{k+1}-z_k\|_\omega^2}{2\,(y_{k+1}-y_k)^\top K\,(x_{k+1}-x_k)}$,
    which is enforced by the computed $\bar\eta$.

## Module 2 — Adaptive restarts

Definitions

- Normalized duality gap at epoch $n$ and point $z=(x,y)$ for radius $r>0$
  - $\displaystyle \rho_r^n(z) := \frac{1}{r}\ \max_{(\hat x,\hat y)\in \{\hat z\in Z: \|\hat z - z\|_{\omega_n}\le r\}}\ \{\,L(x,\hat y) - L(\hat x, y)\,\}$.
- Shorthand using a reference point $z_{\mathrm{ref}}$
  - $\displaystyle \mu_n(z, z_{\mathrm{ref}}) := \rho^n_{\ \|z - z_{\mathrm{ref}}\|_{\omega_n}}(z)$.

Restart candidate selection

- Choose between the newest and the averaged point
  - $\displaystyle z_{n,t+1}^c \leftarrow \mathrm{GetRestartCandidate}(z_{n,t+1},\ \bar z_{n,t+1},\ z_{n,0}) = \begin{cases}
    z_{n,t+1}, & \mu_n(z_{n,t+1}, z_{n,0}) < \mu_n(\bar z_{n,t+1}, z_{n,0}),\\
    \bar z_{n,t+1}, & \text{otherwise.}
    \end{cases}$

Restart trigger (defaults: $\beta_{\text{sufficient}}=0.9$, $\beta_{\text{necessary}}=0.1$, $\beta_{\text{artificial}}=0.5$)

- Restart if any holds
  1. Sufficient decay: $\ \mu_n(z_{n,t+1}^c,\ z_{n,0}) \le \beta_{\text{sufficient}}\,\mu_n(z_{n,0},\ z_{n-1,0})$.
  2. Necessary decay with no local progress: $\ \mu_n(z_{n,t+1}^c,\ z_{n,0}) \le \beta_{\text{necessary}}\,\mu_n(z_{n,0},\ z_{n-1,0})$ and $\ \mu_n(z_{n,t+1}^c,\ z_{n,0}) > \mu_n(z_{n,t}^c,\ z_{n,0})$.
  3. Long inner loop: $\ t \ge \beta_{\text{artificial}}\, k$.

## Module 3 — Primal weight updates

Initialization

- $\displaystyle \omega_0 \leftarrow \mathrm{InitializePrimalWeight}(c,q) := \begin{cases}
  \|c\|_2/\|q\|_2, & \|q\|_2 > \varepsilon_{\text{zero}},\\
  1, & \text{otherwise.}
  \end{cases}$

Update at each restart epoch $n\ge 1$

- Let $\Delta_x^n := \|x_{n,0} - x_{n-1,0}\|_2$ and $\Delta_y^n := \|y_{n,0} - y_{n-1,0}\|_2$.
- If both $\Delta_x^n$ and $\Delta_y^n$ exceed $\varepsilon_{\text{zero}}$, set
  - $\displaystyle \omega_n \leftarrow \exp\Big(\theta\,\log(\Delta_y^n/\Delta_x^n) + (1-\theta)\,\log(\omega_{n-1})\Big)$ with $\theta\in[0,1]$ (PDLP uses $\theta=0.5$).
- Else keep $\omega_n \leftarrow \omega_{n-1}$.

Heuristic aim

- Choose $\omega_n$ to balance weighted distances: approximately $\|(x_{n,t}-x^\star,0)\|_{\omega_n} \approx \|(0,y_{n,t}-y^\star)\|_{\omega_n}$. The unknown ratio is estimated by $\Delta_y^n/\Delta_x^n$ and stabilized in log-scale.

## Termination criteria for approximate optimality

Terminate when there exist $x\in X$, $y\in Y$, and $\lambda\in\Lambda$ with

- Duality gap: $\ \big|\,q^\top y + l^\top \lambda^{+} - u^\top \lambda^{-} - c^\top x\,\big| \le \varepsilon\,\big(1 + |q^\top y + l^\top \lambda^{+} - u^\top \lambda^{-}| + |c^\top x|\big)$.
- Primal feasibility (residual): $\ \left\|\begin{pmatrix} Ax-b \\ (h-Gx)_{+} \end{pmatrix}\right\|_2 \le \varepsilon\,(1+\|q\|_2)$.
- Dual feasibility (reduced-cost residual): $\ \|\,c - K^\top y - \lambda\,\|_2 \le \varepsilon\,(1+\|c\|_2)$.
- For checking, compute $\lambda = \operatorname{proj}_\Lambda(c - K^\top y)$. Typical tolerances: $\varepsilon=10^{-8}$ (high accuracy) or $10^{-4}$ (moderate).

## Minimal symbol dictionary

- Variables: $x\in\mathbb{R}^n$, $y\in\mathbb{R}^{m_1+m_2}$, $z=(x,y)$; $X=\{l\le x\le u\}$; $Y=\{y: y_{1:m_1}\ge 0\}$; $Z=X\times Y$.
- Data: $G\in\mathbb{R}^{m_1\times n}$, $A\in\mathbb{R}^{m_2\times n}$, $K^\top=(G^\top,A^\top)$, $c\in\mathbb{R}^n$, $h\in\mathbb{R}^{m_1}$, $b\in\mathbb{R}^{m_2}$, $q^\top=(h^\top,b^\top)$.
- Dual domain: $\Lambda=\Lambda_1\times\cdots\times\Lambda_n$ with
  $\ \Lambda_i=\{0\}$ if $l_i=-\infty, u_i=\infty$; $\Lambda_i=\mathbb{R}_{-}$ if $l_i=-\infty, u_i\in\mathbb{R}$; $\Lambda_i=\mathbb{R}_{+}$ if $l_i\in\mathbb{R}, u_i=\infty$; otherwise $\Lambda_i=\mathbb{R}$.
- Step parameters: $\eta>0$, $\omega>0$; $\tau=\eta/\omega$, $\sigma=\omega\eta$; $\|\cdot\|_\omega$ as defined above.
- Counters: inner $t=0,1,2,\dots$; outer (restart) $n=0,1,2,\dots$; total iterations $k$.
- Projections: $\operatorname{proj}_X$, $\operatorname{proj}_Y$, $\operatorname{proj}_\Lambda$ are Euclidean projections.

## Module interfaces (for implementation)

- AdaptiveStepOfPDHG$(z,\omega,\hat\eta,k)\ \to\ (z^{+},\eta_{\text{accepted}},\hat\eta_{\text{next}})$ — one safeguarded PDHG step (Module 1).
- GetRestartCandidate$(z_{\text{new}},\bar z,z_{\text{start}})\ \to\ z^{c}$ — candidate selection by normalized duality gap (Module 2).
- PrimalWeightUpdate$(z_{n,0},z_{n-1,0},\omega_{n-1})\ \to\ \omega_n$ — epoch-wise smoothed update (Module 3).
