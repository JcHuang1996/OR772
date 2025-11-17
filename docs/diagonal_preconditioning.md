# Diagonal Preconditioning (PDLP Sec. 3.5) — Two Algorithms

This file gives two diagonal preconditioning procedures—**Pock–Chambolle (PC)** and **Ruiz (ℓ∞‑equilibration)**—and shows how to rescale the LP data \((A,G,c,b,h,u,l)\) consistently. We treat \(K\in\mathbb{R}^{m\times n}\) as the stacked constraint matrix collecting rows from \(A\) and \(G\). The goal is to compute positive diagonal matrices \(D_1\in\mathbb{R}^{m\times m}\) and \(D_2\in\mathbb{R}^{n\times n}\) and form
\[
\tilde K = D_1\, K\, D_2,
\]
then **replace** the LP data by
\[
\tilde A,\tilde G \text{ (the corresponding blocks of }\tilde K),\quad
\hat x = D_2^{-1}x,\quad \tilde c = D_2 c,\quad
(\tilde b,\tilde h)=D_1(b,h),\quad
\tilde u = D_2^{-1}u,\quad \tilde l = D_2^{-1}l.
\]
---

## Algorithm 1 — Pock–Chambolle (PC) Diagonal Preconditioning

**Input.** Matrix \(K\in\mathbb{R}^{m\times n}\); LP data \((A,G,c,b,h,u,l)\); parameter \(\alpha>0\) (PDLP uses \(\alpha=1\)); safeguard \(\varepsilon>0\).  
**Output.** Diagonal \(D_1\in\mathbb{R}^{m\times m}\), \(D_2\in\mathbb{R}^{n\times n}\); rescaled LP data \((\tilde A,\tilde G,\tilde c,\tilde b,\tilde h,\tilde u,\tilde l)\).

**Procedure.**  
1. Compute row and column norms of \(K\):
   \[
   r_i \leftarrow \|K_{i,\cdot}\|_2,\quad i=1,\dots,m;\qquad
   c_j \leftarrow \|K_{\cdot,j}\|_2,\quad j=1,\dots,n.
   \]
2. Set diagonal scalings (guard zeros with \(\varepsilon\)):
   \[
   (D_1)_{ii} \leftarrow \big(\max\{r_i,\varepsilon\}\big)^{-\alpha/2},\qquad
   (D_2)_{jj} \leftarrow \big(\max\{c_j,\varepsilon\}\big)^{\alpha/2}.
   \]
   (This matches the PC family \( (D_1)_{jj}=\|K_{j,\cdot}\|_2^{-\alpha/2},\; (D_2)_{ii}=\|K_{\cdot,i}\|_2^{\alpha/2}\).) 
3. Form \(\tilde K \leftarrow D_1 K D_2\).  
4. Update LP data using the rule at the top to obtain \((\tilde A,\tilde G,\tilde c,\tilde b,\tilde h,\tilde u,\tilde l)\). 

---

## Algorithm 2 — Ruiz (ℓ∞‑Equilibration) Diagonal Preconditioning

**Input.** Matrix \(K\in\mathbb{R}^{m\times n}\); LP data \((A,G,c,b,h,u,l)\); iteration cap \(T\in\mathbb{N}\) (PDLP uses \(T=10\)); tolerance \(\mathrm{tol}>0\); safeguard \(\varepsilon>0\).  
**Output.** Diagonal \(D_1,D_2\) such that row/column \(\ell_\infty\)-norms of \(\tilde K=D_1 K D_2\) are (approximately) balanced; rescaled LP data \((\tilde A,\tilde G,\tilde c,\tilde b,\tilde h,\tilde u,\tilde l)\).

**Procedure.**  
1. Initialize \(D_1\leftarrow I_m\), \(D_2\leftarrow I_n\), \(K'\leftarrow K\).
2. For \(t=1,\dots,T\) (or until convergence):
   - Compute guarded \(\ell_\infty\)-norms: 
     \[
     r_i\leftarrow \max\{\|K'_{i,\cdot}\|_\infty,\varepsilon\},\quad
     c_j\leftarrow \max\{\|K'_{\cdot,j}\|_\infty,\varepsilon\}.
     \]
   - Build stage scalings \(S_1\gets\mathrm{diag}(1/\sqrt{r_1},\dots,1/\sqrt{r_m})\), \(S_2\gets\mathrm{diag}(1/\sqrt{c_1},\dots,1/\sqrt{c_n})\).  
   - Update \(K'\leftarrow S_1 K' S_2\); accumulate \(D_1\leftarrow S_1 D_1\), \(D_2\leftarrow D_2 S_2\).  
   - Stop if both row and column \(\ell_\infty\)-norms of \(K'\) are within \(\mathrm{tol}\) of 1.
   (One iteration corresponds to \( (D_1)_{ii}=\sqrt{\|K_{i,\cdot}\|_\infty}\), \( (D_2)_{jj}=\sqrt{\|K_{\cdot,j}\|_\infty}\).) 【fileciteturn2file3L42-L52】
3. Set \(\tilde K\leftarrow K'\) and output \(D_1,D_2\).  
4. Update LP data using the rule at the top to obtain \((\tilde A,\tilde G,\tilde c,\tilde b,\tilde h,\tilde u,\tilde l)\). 

**Remark.** PDLP’s default is **10 Ruiz iterations followed by one PC scaling step**, i.e., apply Algorithm 2 then Algorithm 1.
