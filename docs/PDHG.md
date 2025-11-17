## PDHG

### **Problem Statement**

Solve the saddle-point problem

$\min_{x \in X} \max_{y \in Y} L(x, y) = c^\top x - y^\top Kx + q^\top y$,

$s.t. K^\top = [G^\top, A^\top], q^\top = [h^\top, b^\top]$,

$X = \{x \in \mathbb{R}^n \mid l \le x \le u\}, \quad Y = \{y \in \mathbb{R}^{m_1+m_2} \mid y_{1:m_1} \ge 0\}$.

corresponding to the primal & dual pair:

$$\begin{aligned} \min_x \;& c^\top x & \text{s.t. } Gx \ge h,~ Ax=b,~ l \le x \le u,\\ \max_y \;& q^\top y & \text{s.t. } K^\top y = c,~ y_{1:m_1} \ge 0. \end{aligned}$$

### **Assumptions & Required Operators**

projection operators:

- **Projection X:**

  $\displaystyle \text{proj}_X(x) = \min(\max(x, l), u)$

- **Projection onto Y:**

  $\displaystyle \text{proj}_Y(y) = [\,\max(y{1:m_1}, 0),~ y_{m_1+1:m_1+m_2}\,]$

### **Algorithm Specification**

Given $ x^0, y^0$, and step-size parameters $\tau>0, \sigma>0$ satisfying

$\tau\sigma\|K\|_2^2 \le 1$,

**Iterate for** $k = 0, 1, 2, \dots, T-1$:
$$
\begin{aligned} x^{k+1} &= \text{proj}_X\!\left(x^k - \tau (c - K^\top y^k)\right),\\[4pt] y^{k+1} &= \text{proj}_Y\!\left(y^k + \sigma (q - K(2x^{k+1} - x^k))\right). \end{aligned}
$$
Terminate when user-defined convergence test holds (e.g. primal/dual residuals or iteration limit).

### **User-Defined Parameters**

- **Primal weight (**$\omega$**)** — balances update scales between primal and dual.

- **Step sizes:**

  $\tau, \sigma > 0, with\quad \tau\sigma\|K\|_2^2 \le 1$.

  Actually, all decided by primal weight $\omega$ and one single parameter $\eta$:

  $\tau = \eta / \omega, \ \sigma = \omega \eta$

  where $ \eta \le 1/\|K\|_2$.

- For the baseline PDHG algorithm that we use for comparisons, we consider two simple choices for $\eta$ and $ω$. For the step size, set $η = 0.9/∥K∥_2$ where $∥K∥_2$ is estimated via power iteration, and for the primal

  weight we set $ω$ = 1; this is similar to the default parameters in the standard PDHG implementation.

