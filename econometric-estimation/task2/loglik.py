# Purpose:    Implement normal-mixture likelihood calculations directly rather
#             than using a pre-built mixture package. The required K=2
#             interface remains explicit, while shared component-array helpers
#             support a K=3 diagnostic comparison without duplicating the
#             likelihood formula.
# Inputs:     NumPy arrays of observations and scalar model parameters.
# Outputs:    Log densities, total log-likelihoods, mixture CDF values, and
#             conversions between reported and optimizer parameterizations.
# Key Steps:  Compute weighted component log densities -> combine them with
#             log-sum-exp -> sum across observations -> transform constrained
#             parameters for K=2 and generic-K optimizers.
# How to Run: Not run directly; imported by `estimate.py`, `estimate_k3.py`, and
#             `diagnostics.py`.

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

FloatArray = np.ndarray
ModelParameters = tuple[float, float, float, float, float]
ComponentParameters = tuple[FloatArray, FloatArray, FloatArray]


def mixture_logpdf_components(
    x: FloatArray,
    weights: FloatArray,
    means: FloatArray,
    sigmas: FloatArray,
) -> FloatArray:
    """Return one stable generic-K mixture log density per observation.

    A shared array implementation lets K=2 and K=3 use exactly the same likelihood formula.
    """
    # An extremely negative optimizer logit can underflow to a numerical zero
    # weight; log(0) = -inf correctly removes that negligible component.
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    component_logs = log_weights[None, :] + norm.logpdf(
        x[:, None],
        loc=means[None, :],
        scale=sigmas[None, :],
    )
    return np.asarray(logsumexp(component_logs, axis=1))


def mixture_logpdf(
    x: FloatArray,
    pi: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> FloatArray:
    """Return one stable K=2 mixture log density per observation.

    Log-sum-exp avoids underflow while keeping the required mixture formula explicit.
    """
    return mixture_logpdf_components(
        x,
        np.array([pi, 1.0 - pi]),
        np.array([mu1, mu2]),
        np.array([sigma1, sigma2]),
    )


def loglik(
    x: FloatArray,
    pi: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> float:
    """Sum observation-level log densities into the maximum-likelihood objective.

    A log sum is numerically safer than multiplying 10,000 small density values.
    """
    return float(np.sum(mixture_logpdf(x, pi, mu1, sigma1, mu2, sigma2)))


def mixture_cdf(
    x: FloatArray,
    pi: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> FloatArray:
    """Return the K=2 mixture CDF for quantile diagnostics.

    Weighting component CDFs gives the fitted cumulative probability at every x.
    """
    return np.asarray(
        pi * norm.cdf(x, loc=mu1, scale=sigma1)
        + (1.0 - pi) * norm.cdf(x, loc=mu2, scale=sigma2)
    )


def mixture_cdf_components(
    x: FloatArray,
    weights: FloatArray,
    means: FloatArray,
    sigmas: FloatArray,
) -> FloatArray:
    """Return the weighted generic-K component CDF.

    This supports K=3 graphical diagnostics while preserving the explicit K=2 interface.
    """
    component_cdfs = norm.cdf(
        x[:, None],
        loc=means[None, :],
        scale=sigmas[None, :],
    )
    return np.asarray(component_cdfs @ weights)


def to_unconstrained(
    pi: float,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> FloatArray:
    """Map reported K=2 parameters to unrestricted optimizer coordinates.

    Log sigmas and a weight logit allow unconstrained search while preserving valid parameters.
    """
    logit_pi = np.log(pi) - np.log1p(-pi)
    return np.array([mu1, mu2, np.log(sigma1), np.log(sigma2), logit_pi])


def from_unconstrained(theta: FloatArray) -> ModelParameters:
    """Convert optimizer coordinates back to valid, interpretable K=2 parameters.

    Exponentials enforce positive sigmas and the logistic transform keeps pi between zero and one.
    """
    mu1, mu2 = float(theta[0]), float(theta[1])
    sigma1, sigma2 = float(np.exp(theta[2])), float(np.exp(theta[3]))
    pi = float(expit(theta[4]))
    return pi, mu1, sigma1, mu2, sigma2


def negative_loglik(theta: FloatArray, x: FloatArray) -> float:
    """Return negative log-likelihood for SciPy's minimization interface.

    Minimizing this objective is equivalent to maximizing the required log-likelihood.
    """
    return -loglik(x, *from_unconstrained(theta))


def components_to_unconstrained(
    weights: FloatArray,
    means: FloatArray,
    sigmas: FloatArray,
) -> FloatArray:
    """Map generic-K parameters to means, log sigmas, and K-1 relative logits.

    Using the first weight as a baseline removes the adding-up redundancy.
    """
    relative_logits = np.log(weights[1:]) - np.log(weights[0])
    return np.concatenate([means, np.log(sigmas), relative_logits])


def components_from_unconstrained(theta: FloatArray, k: int) -> ComponentParameters:
    """Convert generic optimizer coordinates to valid component arrays.

    Exponentials enforce positive sigmas and softmax enforces simplex weights.
    """
    means = theta[:k]
    sigmas = np.exp(theta[k:2 * k])
    logits = np.concatenate([[0.0], theta[2 * k:]])
    logits -= logits.max()
    weights = np.exp(logits) / np.exp(logits).sum()
    return np.asarray(weights), np.asarray(means), np.asarray(sigmas)


def components_negative_loglik(theta: FloatArray, x: FloatArray, k: int) -> float:
    """Return generic-K negative log-likelihood for numerical minimization.

    The common objective ensures the K=3 comparison differs only in component count.
    """
    weights, means, sigmas = components_from_unconstrained(theta, k)
    return -float(np.sum(mixture_logpdf_components(x, weights, means, sigmas)))
