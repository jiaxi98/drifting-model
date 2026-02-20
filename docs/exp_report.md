# Experiment report for paper: Generative Modeling via Drifting
We have done some preliminary experiments over MNIST and CIFAR10 with an extended implementation ([link](https://github.com/jiaxi98/drifting-model)). Unfortunately, we are not able to reproduce the results shown in the paper and the FID score looks very unsatisfactory.

## Batch size of positive and negative samples
We investigate different positive and negative batch sizes, e.g. `batch_n_pos, batch_n_neg` parameters in config which is used to evaluate the drift field via Monte Carlo. Intuitively speaking, with larger batch size, the estimation of drift field should be less noisy and hopefully the training will benefit. Notice their implementation (Alg 2) uses a biased version of the Monte Carlo estimator and they claim this choice performs better.


## Kernel choices
We investigate different kernel choices. The paper uses L2 kernel, while we found that Gaussian kernel has theoretical advantages: it provides an interpretation of the drift field loss function as an approximation to the Fisher information. And the resulting gradient from the stop gradient objective can be viewed as the Wasserstein gradient flow of the KL divergence between the generated distribution and real data distribution, projected via NTK.
### Implementation

### L2 kernel
### Gaussian kernel


## Feature extractor
We believe the powerful feature extractor is probably the key to the success of the drift model, although we do not reproduce the results with the same extractor.

