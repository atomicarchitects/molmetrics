from typing import Dict, Sequence, Tuple
import functools
import collections

import jax
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn

from molmetrics.datatypes import LocalEnvironment, Atom


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
    positions = [atom.position for atom in env.neighbors]
    positions = jnp.asarray(positions)
    return np.asarray(bispectrum(positions, lmax))


class BispectraSamples:
    """Represents a collection of bispectrum samples for local environments."""
    
    def __init__(self, samples: Dict[LocalEnvironment, np.ndarray]) -> None:
        self.samples = samples
    
    def aggregate_by_local_environment(self) -> Dict[LocalEnvironment, np.ndarray]:
        """Aggregates samples by local environment."""
        samples = collections.defaultdict(list)
        for env, sample in self.samples.items():
            # Remove positions from atoms.
            env_no_positions = LocalEnvironment(env.central_atom, tuple([Atom(atom.symbol) for atom in env.neighbors]))
            samples[env_no_positions].append(sample)
        
        return jax.tree_map(np.stack, samples)