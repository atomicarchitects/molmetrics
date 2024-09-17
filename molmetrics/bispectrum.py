from typing import Dict, Sequence, Tuple
import collections
import functools

import jax
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn
from rdkit import Chem

from molmetrics.datatypes import LocalEnvironment

@functools.partial(jax.jit, static_argnames=("lmax",))
def bispectrum(positions: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """Computes the bispectrum of a set of (relative) positions."""
    assert positions.shape == (positions.shape[0], 3)
    x = e3nn.sum(
        e3nn.s2_dirac(positions, lmax=lmax, p_val=1, p_arg=-1), axis=0
    )
    rtp = e3nn.reduced_symmetric_tensor_product_basis(
        x.irreps, 3, keep_ir=["0e", "0o"],
        _use_optimized_implementation=True
    )
    return jnp.einsum("ijkz,i,j,k->z", rtp.array, x.array, x.array, x.array)


def compute_bispectrum_for_local_environment(env: LocalEnvironment, lmax: int) -> np.ndarray:
    """Computes the bispectrum of a local environment."""
    positions = jnp.asarray(env.neighbor_positions)
    return np.asarray(bispectrum(positions, lmax))