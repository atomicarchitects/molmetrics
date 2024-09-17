"""
Helpers for computing the Maximum Mean Discrepancy (MMD) distance between two sets of samples.

Reference: https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
Adapted from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
"""

from typing import Any, Dict
import functools

import jax
import jax.numpy as jnp
import chex


@functools.partial(
    jax.jit, static_argnames=("batch_size", "num_batches", "num_kernels")
)
def compute_maximum_mean_discrepancy(
    source_samples: chex.Array,
    target_samples: chex.Array,
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
    num_kernels: int = 30,
) -> float:
    """
    Calculate the maximum mean discrepancy distance between two lists of samples.
    """

    def rbf_kernel(X: chex.Array, Y: chex.Array, gamma: float) -> chex.Array:
        """RBF (Gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2))"""
        return jnp.exp(
            -gamma * (jnp.linalg.norm(X[:, None] - Y[None, :], axis=-1) ** 2)
        )

    def mmd_rbf(X: chex.Array, Y: chex.Array, gammas: chex.Array) -> float:
        """MMD using RBF (Gaussian) kernel."""

        def squared_mmd_rbf_kernel(gamma: float) -> float:
            XX = rbf_kernel(X, X, gamma).mean()
            YY = rbf_kernel(Y, Y, gamma).mean()
            XY = rbf_kernel(X, Y, gamma).mean()
            return jnp.abs(XX + YY - 2 * XY)

        return jnp.sqrt(jax.vmap(squared_mmd_rbf_kernel)(gammas).sum())

    def mmd_rbf_batched(
        X: chex.Array, Y: chex.Array, gammas: chex.Array, rng: chex.PRNGKey
    ) -> float:
        """Helper function to compute MMD in batches."""
        X_rng, Y_rng = jax.random.split(rng)
        X_indices = jax.random.randint(
            X_rng, shape=(batch_size,), minval=0, maxval=len(X)
        )
        Y_indices = jax.random.randint(
            Y_rng, shape=(batch_size,), minval=0, maxval=len(Y)
        )
        X_batch, Y_batch = X[X_indices], Y[Y_indices]
        return mmd_rbf(X_batch, Y_batch, gammas)

    X = jnp.asarray(source_samples)
    if len(X.shape) == 1:
        X = X[:, None]

    Y = jnp.asarray(target_samples)
    if len(Y.shape) == 1:
        Y = Y[:, None]

    if batch_size is None:
        batch_size = min(len(X), len(Y))

    # We can only compute the MMD if the number of features is the same.
    assert X.shape[1] == Y.shape[1]

    # We set the kernel widths uniform in logspace.
    gammas = jnp.logspace(-3, 3, num_kernels)

    return jax.vmap(lambda rng: mmd_rbf_batched(X, Y, gammas, rng))(
        jax.random.split(rng, num_batches)
    ).mean()


def compute_maximum_mean_discrepancies(
    source_samples_dict: Dict[Any, jnp.ndarray],
    target_samples_dict: Dict[Any, jnp.ndarray],
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
) -> Dict[Any, float]:
    """
    Compute the maximum mean discrepancy distance for each key in the source and target dictionaries.
    """
    results = {}
    for key in source_samples_dict:
        if key not in target_samples_dict:
            continue

        mmd_rng, rng = jax.random.split(rng)
        results[key] = compute_maximum_mean_discrepancy(
            source_samples_dict[key],
            target_samples_dict[key],
            mmd_rng,
            batch_size,
            num_batches,
        )

    return results
