"""
Module for the greedy (deterministic argmax) sampling strategy.
"""

from abc import ABC

import jax.numpy as np

from swarmrl.sampling_strategies.sampling_strategy import DiscreteSamplingStrategy


class GreedyDistribution(DiscreteSamplingStrategy, ABC):
    """
    Deterministic discrete strategy: always pick the highest-logit action.

    Unlike :class:`GumbelDistribution` (which adds Gumbel noise to draw a categorical
    sample) this takes the argmax of the logits, i.e. the greedy / most-probable action.
    Intended for deployment/inference where a fixed optimal policy is wanted rather than
    the entropic exploring policy used during training.
    """

    def __call__(self, logits: np.ndarray, rng_key=None) -> np.ndarray:
        """
        Select the greedy action for every colloid.

        Parameters
        ----------
        logits : np.ndarray (n_colloids, n_dimensions)
                Logits from the model to use in the computation for all colloids.
        rng_key : Optional[jax.Array]
                Unused; accepted so the signature matches the stochastic strategies.

        Returns
        -------
        indices : np.ndarray (n_colloids,)
                Index of the highest-logit action for each colloid.
        """
        return np.argmax(logits, axis=-1)
