# LSEBCE — Gradient & Hessian Derivation

**LSEBCE** = LogSumExp Binary Cross Entropy, a smooth multiple-instance learning (MIL) objective. The bag-level logit is a LogSumExp pooling of instance logits, followed by a sigmoid and binary cross-entropy loss.

---

## 1. Forward pass

Let $\{z_j\}_{j=1}^{n}$ be per-instance logits in a bag, and $t \in \{0,1\}$ the bag label.

| Step | Expression |
|------|-----------|
| Weighted exponentials | $s = \sum_j e^{r z_j}$ |
| Bag logit (LSE pooling) | $y = \frac{1}{r} \log s$ |
| Bag probability | $p = \sigma(y) = \frac{1}{1 + e^{-y}}$ |
| Binary cross-entropy | $L = -\,[\,t \log p + (1-t) \log(1-p)\,]$ |

The softmax weight for instance $j$ is:

$$
w_j \equiv \frac{\partial y}{\partial z_j} = \frac{1}{r}\cdot\frac{1}{s}\cdot r e^{r z_j} = \frac{e^{r z_j}}{s}.
$$

---

## 2. Gradient $\displaystyle\frac{\partial L}{\partial z_j}$

Chain rule:

$$
\frac{\partial L}{\partial z_j}
= \underbrace{\frac{\partial L}{\partial p}}_{\text{(i)}} \;
  \underbrace{\frac{\partial p}{\partial y}}_{\text{(ii)}} \;
  \underbrace{\frac{\partial y}{\partial z_j}}_{\text{(iii)}} .
$$

**(i)** Cross-entropy derivative:

$$
\frac{\partial L}{\partial p}
= -\frac{t}{p} + \frac{1-t}{1-p}
= \frac{p - t}{p(1-p)} .
$$

**(ii)** Sigmoid derivative:

$$
\frac{\partial p}{\partial y} = p(1-p).
$$

**(iii)** LSE derivative (defined above):

$$
\frac{\partial y}{\partial z_j} = w_j .
$$

**Product** simplifies because (i) × (ii) collapses to $p - t$:

$$
\boxed{\frac{\partial L}{\partial z_j} = (p - t)\, w_j}.
$$

---

## 3. Hessian $\displaystyle\frac{\partial^2 L}{\partial z_j^2}$

Differentiate the gradient using the product rule:

$$
\frac{\partial^2 L}{\partial z_j^2}
= \underbrace{\frac{\partial(p - t)}{\partial z_j}}_{A}\; w_j
\;+\; (p - t)\; \underbrace{\frac{\partial w_j}{\partial z_j}}_{B}.
$$

### Term $A$

$$
\frac{\partial(p - t)}{\partial z_j}
= \frac{\partial p}{\partial y} \frac{\partial y}{\partial z_j}
= p(1-p)\, w_j .
$$

### Term $B$

$$
\begin{aligned}
\frac{\partial w_j}{\partial z_j}
&= \frac{\partial}{\partial z_j} \frac{e^{r z_j}}{s} \\[4pt]
&= \frac{r e^{r z_j} s - e^{r z_j} \cdot r e^{r z_j}}{s^2} \\[4pt]
&= r w_j - r w_j^2 \\[4pt]
&= r\, w_j (1 - w_j).
\end{aligned}
$$

### Combined

$$
\boxed{\frac{\partial^2 L}{\partial z_j^2}
= p(1-p)\, w_j^2 \;+\; r\,(p - t)\, w_j (1 - w_j)}.
$$

---

## 4. Remarks

- **Gradient** $(p - t) w_j$: the residual $(p - t)$ is scaled by the softmax weight $w_j$, so instances with larger $e^{r z_j}$ (more "positive" evidence) receive larger gradient updates.
- **Hessian** has two terms:
  1. $p(1-p) w_j^2$ — the standard BCE Hessian scaled by $w_j^2$, always non-negative.
  2. $r\,(p - t) w_j (1 - w_j)$ — the MIL-specific correction. It can be negative when $p < t$ and $w_j$ is small, reflecting that the instance is uncertain evidence for the positive label. The implementation clips the Hessian at $10^{-4}$ for numerical safety.
- The gradient and Hessian above are *per instance*. In gradient boosting they are typically aggregated over all instances in all bags.

---

## 5. Correspondence to implementation (`src/milgboost/objective/lse.py`)

```python
w_ij = exp_preds / (bag_sum_exp[bag_ids] + 1e-12)    # w_j
bag_y = (1.0 / r) * np.log(bag_sum_exp + 1e-12)      # y
bag_p = 1.0 / (1.0 + np.exp(-bag_y))                  # p
p_i   = bag_p[bag_ids]
t_i   = y

grad  = (p_i - t_i) * w_ij                            # (p - t) w
hess  = p_i * (1.0 - p_i) * (w_ij**2)                 # p(1-p) w²
hess += r * (p_i - t_i) * w_ij * (1.0 - w_ij)         # r(p-t) w(1-w)
hess  = np.clip(hess, 1e-4, None)                     # clip for safety
```
