import warnings
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.neighbors import RadiusNeighborsRegressor


def _get_flat_weights(dist, weights, flat_size):
    """Get the weights from an array of distances and a parameter ``weights``.

    Parameters
    ----------
    dist : ndarray
        The input distances.

    weights : {'uniform', 'distance' or a callable}
        The kind of weighting used.

    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        If ``weights == 'uniform'``, then returns None.
    """

    if weights in ("uniform", None):
        weights = np.ones(flat_size, dtype=bool)
    else:
        if weights == "distance":
            weights = dist
        elif callable(weights):
            weights = weights(dist)
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )
        weights = np.concatenate(weights, axis=0)

    return weights


class FastRadiusRegressor(RadiusNeighborsRegressor):
    def _get_sparse_weights(
        self,
        training_size: int,
        neigh_dist: Sequence[np.ndarray],
        neigh_ind: Sequence[np.ndarray],
    ) -> sparse.csr_matrix:
        """
        Parameters
        ----------
        training_size : int
            Number of training samples.
        neigh_dist : Sequence[np.ndarray]
            List of neighbors distances.
        neigh_ind : Sequence[np.ndarray]
            List of neighbors indices.
        Returns
        -------
        sparse.csr_matrix
            Output weight matrix.
        """
        dist_zero_constant = 1e10  # high value number to emulate identity function when dist == 0.0

        size = len(neigh_ind)
        lengths = np.asarray([len(ind) for ind in neigh_ind])

        neigh_dst = np.concatenate(neigh_ind, axis=0)
        neigh_src = np.repeat(np.arange(size), repeats=lengths)

        weights = _get_flat_weights(neigh_dist, self.weights, len(neigh_src))
        weights = sparse.csr_matrix(
            (weights, (neigh_src, neigh_dst)), shape=(size, training_size)
        )

        # invert distance and assign binary encoding to point with zero distance to training points
        if self.weights == "distance":
            weights.data = np.where(
                weights.data == 0.0, dist_zero_constant, 1.0 / weights.data
            )

        return weights

    def predict(self, X):
        """Predict the target for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), \
                dtype=double
            Target values.
        """
        neigh_dist, neigh_ind = self.radius_neighbors(X)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        weights = self._get_sparse_weights(len(_y), neigh_dist, neigh_ind)

        norm_factor = weights.sum(axis=1)
        y_pred = weights @ _y
        with np.errstate(divide="ignore"):
            y_pred = np.where(norm_factor > 0, y_pred / norm_factor, np.nan)

        if np.any(np.isnan(y_pred)):
            empty_warning_msg = (
                "One or more samples have no neighbors "
                "within specified radius; predicting NaN."
            )
            warnings.warn(empty_warning_msg)

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred
