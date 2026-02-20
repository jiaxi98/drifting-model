# Drift Field Numerical Derivation and Monte Carlo Estimation

This note summarizes how to compute the drifting field numerically (matching Algorithm 2 in the paper) and how to interpret Monte Carlo behavior when $y^+$ and $y^-$ come from the same distribution.

## 1. Setup

Let

- $x \in \mathbb{R}^{N \times D}$: query/generated features
- $y^+ \in \mathbb{R}^{N_{+} \times D}$: positive samples
- $y^- \in \mathbb{R}^{N_{-} \times D}$: negative samples

For one temperature $\tau$:

$$
d^+_{ij} = \|x_i - y^+_j\|_2,\quad
d^-_{ik} = \|x_i - y^-_k\|_2.
$$

Logits:

$$
\ell^+_{ij} = -\frac{d^+_{ij}}{\tau},\quad
\ell^-_{ik} = -\frac{d^-_{ik}}{\tau}.
$$

Concatenate along sample axis:

$$
\ell = [\ell^+, \ell^-] \in \mathbb{R}^{N \times (N_+ + N_-)}.
$$

## 2. Two-axis normalization used in practice

The implementation uses two softmaxes on the same logit matrix:

$$
A^{\text{row}}_{im} = \frac{e^{\ell_{im}}}{\sum_{m'} e^{\ell_{im'}}},
\quad
A^{\text{col}}_{im} = \frac{e^{\ell_{im}}}{\sum_{i'} e^{\ell_{i'm}}}.
$$

Then combine them by geometric mean:

$$
A_{im} = \sqrt{A^{\text{row}}_{im}A^{\text{col}}_{im}}.
$$

Split $A$ back into

$$
A^+ \in \mathbb{R}^{N \times N_+},\quad
A^- \in \mathbb{R}^{N \times N_-}.
$$

This is exactly what the code does:

- `A_row = softmax(logit, dim=1)`
- `A_col = softmax(logit, dim=0)`
- `A = sqrt(A_row * A_col)`

## 3. Cross-weighting and drift

Define:

$$
W^+_{ij} = A^+_{ij}\sum_{k=1}^{N_-}A^-_{ik},
\quad
W^-_{ik} = A^-_{ik}\sum_{j=1}^{N_+}A^+_{ij}.
$$

Then the numerical drift estimator is:

$$
V_i =
\sum_{j=1}^{N_+}W^+_{ij}y^+_j
-
\sum_{k=1}^{N_-}W^-_{ik}y^-_k.
$$

In matrix form:

$$
V = W^+y^+ - W^-y^-.
$$

## 4. What to implement for Monte Carlo drift estimation

If we want a Monte Carlo estimate of population drift for fixed query points $x$:

1. Repeat for $t=1,\dots,T$:
   1. Sample i.i.d. mini-batches:
      - $y^{+,t}_j \sim p$
      - $y^{-,t}_k \sim q$
   2. Compute $V^{(t)}(x)$ by the numerical procedure above.
2. Average:

$$
\hat V_{\text{MC}}(x) = \frac{1}{T}\sum_{t=1}^T V^{(t)}(x).
$$

### Minimal implementation skeleton

```python
def estimate_drift_mc(x, sample_pos, sample_neg, tau, T, mask_self=False):
    # x: (N, D), fixed probe points
    v_acc = torch.zeros_like(x)
    for _ in range(T):
        y_pos = sample_pos()  # (N_pos, D), iid from p
        y_neg = sample_neg()  # (N_neg, D), iid from q

        dist_pos = torch.cdist(x, y_pos, p=2)
        dist_neg = torch.cdist(x, y_neg, p=2)
        if mask_self:
            # only when y_neg explicitly contains x in aligned indices
            n = min(dist_neg.shape[0], dist_neg.shape[1])
            dist_neg[:, :n] += torch.eye(n, device=x.device) * 1e6

        logit = torch.cat([-dist_pos / tau, -dist_neg / tau], dim=1)
        a_row = torch.softmax(logit, dim=1)
        a_col = torch.softmax(logit, dim=0)
        a = torch.sqrt(a_row * a_col)
        a_pos, a_neg = a[:, :y_pos.shape[0]], a[:, y_pos.shape[0]:]

        w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
        w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
        v = w_pos @ y_pos - w_neg @ y_neg
        v_acc += v

    return v_acc / T
```

## 5. If $y^+$ and $y^-$ are from the same distribution, should drift be near zero?

Short answer: yes in expectation, not exactly zero for one finite batch.

If $p=q$, the population drift is designed to be zero (equilibrium condition). Intuitively, swapping $y^+$ and $y^-$ flips sign of $(y^+-y^-)$, and the construction is symmetric/anti-symmetric so the expectation cancels.

But numerically with finite Monte Carlo samples:

- one batch gives a noisy estimate, typically not exactly zero
- averaging over more samples (larger batch or larger $T$) should push mean drift toward zero

Typical behavior:

- $\mathbb{E}[\hat V_{\text{MC}}(x)] \approx 0$ under correct implementation and iid sampling
- $\|\hat V_{\text{MC}}(x)\|$ decreases as sampling increases (variance reduction)

If it does not trend to near zero in the $p=q$ sanity check, likely causes are implementation mismatch (normalization, masking, weighting, or feature scaling).
