# Feature Construction (Appendix A.5)

This document explains the exact feature vectors built by
`feature_encoder.extract_feature_fullsets(...)` and maps each one to the
paper's Appendix A.5 definition.

## Scope

- Function: `feature_encoder.py::extract_feature_fullsets`
- Output type: `List[Tensor]`, each tensor has shape `(B, D)`
- Goal: construct all A.5 feature groups
- Not included: A.6 feature/drift normalization (handled elsewhere)

## Notation

- Input batch to generator: `x in R^{B x C x H x W}`
- Encoder input (after optional VAE decode): `x_e in R^{B x C0 x H0 x W0}`
- One feature map from encoder: `F in R^{B x Ci x Hi x Wi}`
- A returned feature vector always has shape `(B, Ci)` (or `(B, C0)` / vanilla `(B, C*H*W)`)

## Returned Features and Formula-to-Code Mapping

### 1. Vanilla drifting term (without phi)

Paper: A.5 says all feature losses are summed "in addition to the vanilla drifting loss without phi".

Definition:

- `v_vanilla = vec(x) in R^{B x (C*H*W)}`

Code:

- Block: `if include_vanilla: features.append(x.flatten(start_dim=1))`

### 2. Encoder-input energy term

Paper: A.5 adds encoder input channel-wise mean of squared values.

Definition:

- `e_c = (1 / (H0*W0)) * sum_{h,w} x_e[c,h,w]^2`
- `e in R^{B x C0}`

Code:

- Block: `features.append((encoder_input ** 2).mean(dim=(2, 3)))`

### 3. (a) Per-location vectors for each feature map

Paper A.5 (a): `Hi*Wi` vectors, one per location, each `Ci`-dim.

Definition:

- `v_{h,w} = F[:, :, h, w] in R^{B x Ci}`
- total vectors from this group: `Hi*Wi`

Code:

- Block labeled `(a)`:
  - `per_location = fmap.permute(0, 2, 3, 1).reshape(B, Hi*Wi, Ci)`
  - iterate `for vec in per_location.unbind(dim=1): features.append(vec)`

### 4. (b) Global mean and global std

Paper A.5 (b): one global mean and one global std per feature map.

Definition:

- `mu = (1 / (Hi*Wi)) * sum_{h,w} F[:, :, h, w] in R^{B x Ci}`
- `sigma = std_{h,w}(F[:, :, h, w]) in R^{B x Ci}`

Code:

- Block labeled `(b)`:
  - `features.append(F.adaptive_avg_pool2d(fmap, 1).flatten(1))`
  - `features.append(fmap.flatten(2).std(dim=2, unbiased=False))`

### 5. (c) Non-overlapping 2x2 patch mean/std vectors

Paper A.5 (c): produce `(Hi/2)*(Wi/2)` mean vectors and same number of std vectors.

Definition:

- For each 2x2 patch `P_{u,v}`:
  - `mu2_{u,v} = mean(P_{u,v}) in R^{B x Ci}`
  - `sigma2_{u,v} = std(P_{u,v}) in R^{B x Ci}`

Code:

- Block labeled `(c)`:
  - crop to multiples of 2: `fmap2 = fmap[:, :, :h2*2, :w2*2]`
  - means: `mean2 = avg_pool2d(fmap2, kernel_size=2, stride=2)`
  - stds: `patch2 = unfold(..., kernel_size=2, stride=2)`, then std over 4 elements
  - flatten patch grid to vectors and append all mean vectors, then all std vectors

### 6. (d) Non-overlapping 4x4 patch mean/std vectors

Paper A.5 (d): produce `(Hi/4)*(Wi/4)` mean vectors and same number of std vectors.

Definition:

- For each 4x4 patch `Q_{u,v}`:
  - `mu4_{u,v} = mean(Q_{u,v}) in R^{B x Ci}`
  - `sigma4_{u,v} = std(Q_{u,v}) in R^{B x Ci}`

Code:

- Block labeled `(d)`:
  - crop to multiples of 4: `fmap4 = fmap[:, :, :h4*4, :w4*4]`
  - means: `mean4 = avg_pool2d(fmap4, kernel_size=4, stride=4)`
  - stds: `patch4 = unfold(..., kernel_size=4, stride=4)`, then std over 16 elements
  - flatten patch grid to vectors and append all mean vectors, then all std vectors

## Output Order

Within `extract_feature_fullsets`, features are appended in this order:

1. vanilla `vec(x)` (if `include_vanilla=True`)
2. input energy `mean(x_e^2)` per channel
3. for each feature map `F`:
4. all per-location vectors
5. global mean
6. global std
7. all 2x2 patch means
8. all 2x2 patch stds
9. all 4x4 patch means
10. all 4x4 patch stds

This order is deterministic and directly matches list append order in code.
