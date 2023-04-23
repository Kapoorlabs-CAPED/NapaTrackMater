import functools
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsRegressor
from tqdm import tqdm

from .fast_radius_regression import FastRadiusRegressor


def outdate_fit(method):
    """Records that model fit must be recomputed"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._fitted = False
        return method(self, *args, **kwargs)

    return wrapper


def update_fit(method):
    """Recompute fit if necessary"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self._fitted:
            self._fit()
        return method(self, *args, **kwargs)

    return wrapper


class FateMapping:
    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        radius: float = 1.0,
        reverse: bool = False,
        sigma: float = 0.0,
        weights: str = "distance",
        heatmap: bool = False,
        n_samples: int = 25,
        bind_to_existing: bool = True,
    ) -> None:
        """
        Simulates a fate map experiment from a set of tracks by interpolating coordinates at each time step.

        Parameters
        ----------
        data : Optional[Union[pd.DataFrame, np.ndarray]], optional
            Dataframe with columns TrackID, t, (z), y, x or 2-dim array with length 4 or 5 on 1-axis
        radius : float, optional
            Interpolation neighboord radius
        reverse : bool, optional
            Indicates reverse time step direction
        sigma : float, optional
            Additive gaussian noise sigma
        weights : str, optional
            Interpolation weighting strategy, by default "distance"
        heatmap : bool, optional
            Accumulate the tracks frequency into a heatmap, by default False
        n_samples : int, optional
            Number of samples per individual coordinate, by default 25
        bind_to_existing : bool, optional
            Binds sample to existing data point at starting time, by default True
        """
        self._base_colnames = [ 'Track ID','t', 'z', 'y', 'x']
        self._spatial_columns = ['z', 'y', 'x','Radius', 'Volume', 'Eccentricity Comp First', 'Eccentricity Comp Second', 'Surface Area', 'Cluster Class']
        self._dynamic_columns = ['Speed', 'Motion_Angle', 'Acceleration', 'Distance_Cell_mask', 'Radial_Angle', 'Cell_Axis_Mask']
        self._label_columns = ['Dividing', 'Number_Dividing', 'Touching Neighbours', 'Nearest Neighbours']
        self.reverse = reverse
        self.radius = radius
        self.data = data
        self.sigma = sigma
        self.weights = weights
        self.heatmap = heatmap
        self.n_samples = n_samples
        self.bind_to_existing = bind_to_existing

    def _validate_data(
        self, value: Union[np.ndarray, pd.DataFrame]
    ) -> pd.DataFrame:
        """Sanity checks the data and converts to df if necessary"""
        if value.ndim != 2:
            raise ValueError(
                f"data must be 2-dim array, found {value.ndim}-dim"
            )
      
            value_spatial = pd.DataFrame(
                value, columns=self._base_colnames[:2] + self._spatial_columns
            )

            value_dynamic = pd.DataFrame(
                value, columns=self._base_colnames[:2] + self._dynamic_columns
            )

        return value_spatial, value_dynamic

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: Optional[pd.DataFrame]) -> None:
        """Sets tracking data"""
        self._fitted = False
        self._models = {}

        if value is None:
            self._data = value
            return

        self._data = self._validate_data(value)
        self._tmin = int(round(self._data["t"].min()))
        self._tmax = int(round(self._data["t"].max()))
        self._tracks_by_time = self._data.groupby("t")

    @property
    def weights(self) -> str:
        return self._weights

    @weights.setter
    def weights(self, value: str) -> None:
        """Neighborhood weighting for interpolation (knn regression)"""
        self._weights = value
        for model in self._models.values():
            model.weights = value

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    @outdate_fit
    def radius(self, value: float) -> None:
        """Neighborhood radius for interpolation (knn regression)"""
        self._radius = value

    @property
    def reverse(self) -> bool:
        return self._reverse

    @reverse.setter
    @outdate_fit
    def reverse(self, value: bool) -> None:
        """Sets interpolation direction"""
        self._reverse = value

    def _fit(self) -> None:
        """Fits a model for each time point"""
        if self._data is None:
            raise ValueError("Data must be set before executing Fate Mapping")

        self._models = {
            t: self._fit_model(t)
            for t in tqdm(self.time_iter(), "Fitting interpolation")
        }
        self._fitted = True

    def _fit_model(self, time: int) -> RadiusNeighborsRegressor:
        """Fits the interpolation model to the given time point"""

        # merge consecutive time points
        df = pd.concat(
            (
                self._tracks_by_time.get_group(time),
                self._tracks_by_time.get_group(time + self.step),
            )
        )

        # find tracks belonging to both of them (connections)
        connected = df.groupby("TrackID").size() > 1
        connected = connected[df["TrackID"]].values

        # compute source (X) and target (Y) pairs from connections
        conn_df = df.loc[connected]
        if conn_df.shape[0] > 0:
            conn_df = conn_df.sort_values(by=["TrackID", "t"])
            values = conn_df[self._spatial_columns].values
            X = values[::2]
            Y = values[1::2]
            if self.reverse:
                X, Y = Y, X
        else:
            X, Y = None, None

        # connect disconnected pairs to their nearest neighbors in the subsequent time point
        split_df = df.loc[np.logical_not(connected)]
        split_df = split_df[split_df["t"] == time][
            self._spatial_columns
        ].values
        next_df = self._tracks_by_time.get_group(time + self.step)[
            self._spatial_columns
        ].values
        if split_df.shape[0] > 0 and next_df.shape[0] > 0:
            nn = KNeighborsTransformer(n_neighbors=1)
            nn.fit(next_df)
            neighbors = nn.kneighbors(
                split_df, return_distance=False
            ).squeeze()

            if X is None:
                X = (split_df,)
                Y = next_df[neighbors]
            else:
                X = np.concatenate((X, split_df), axis=0)
                Y = np.concatenate((Y, next_df[neighbors]), axis=0)

        # build regression model
        return FastRadiusRegressor(
            radius=self.radius,
            weights=self.weights,
            algorithm="kd_tree",
            leaf_size=5,
            n_jobs=8,
        ).fit(X, Y)

    @property
    def step(self) -> int:
        """Time step"""
        return -1 if self.reverse else 1

    def time_iter(
        self, t0: Optional[int] = None, max_length: Optional[int] = None
    ) -> Iterable[int]:
        """Time step iterable, starting at `t0`"""
        if t0 is not None and (t0 < self._tmin or t0 > self._tmax):
            raise ValueError(
                f"time point out of models range {(self._tmin, self._tmax)}"
            )

        if self.reverse:
            if t0 is None:
                t0 = self._tmax

            # computing stop with boundary checking
            tN = self._tmin
            if max_length is not None:
                tN = max(tN, t0 - max_length)

            return range(t0, tN, self.step)
        else:
            if t0 is None:
                t0 = self._tmin

            # computing stop with boundary checking
            tN = self._tmax
            if max_length is not None:
                tN = min(tN, t0 + max_length)

            return range(t0, tN, self.step)

    def _compute_heatmap(self, paths: np.ndarray) -> zarr.Array:
        """Accumulates frequency of `path` hits"""
        shape = np.ceil(paths[:, 1:].max(axis=0)).astype(int) + 1
        heatmap = zarr.zeros(
            shape=shape,
            dtype=np.int32,
            store=zarr.MemoryStore(),
            chunks=(1,) + (len(shape) - 1) * (64,),
        )
        df = self._validate_data(paths)
        for t, group in tqdm(df.groupby("t"), "Computing heatmap"):
            coords = group[self._spatial_columns].round().astype(int)
            coords["w"] = 1
            coords = coords.groupby(
                self._spatial_columns, as_index=False
            ).sum()
            heatmap.vindex[
                (int(round(t)),)
                + tuple(coords[self._spatial_columns].values.T)
            ] = coords["w"]
        return heatmap

    @staticmethod
    def _valid_rows(pos: np.ndarray) -> np.ndarray:
        """Returns mask of rows with no nan values"""
        return np.logical_not(np.any(np.isnan(pos), axis=1))

    def _as_track(self, t: int, pos: np.ndarray) -> np.ndarray:
        """Converts coordinates and time to tracks format"""
        t = np.full(pos.shape[0], t)[:, np.newaxis]
        track_ids = np.arange(1, 1 + pos.shape[0])[:, np.newaxis]
        tracks = np.concatenate((track_ids, t, pos), axis=1)
        return tracks[self._valid_rows(pos)]

    def _get_noise_function(self, shape: Tuple[int]) -> Callable:
        """Noise or dummy function given sigma"""
        if self.sigma == 0.0:
            zeros = np.zeros(shape, dtype=np.float32)

            def _fun():
                return zeros

        else:
            rng = np.random.default_rng(42)

            def _fun():
                return rng.normal(scale=self.sigma, size=shape)

        return _fun

    def _knn_sampling(self, coords: np.ndarray) -> np.ndarray:
        """Selects nearest neighbors on reference data"""
        samples = []
        for t in np.unique(coords[:, 0]):
            current = coords[coords[:, 0] == t]
            df = self._tracks_by_time.get_group(int(round(t)))
            X = df[["t"] + self._spatial_columns].values
            nn = KNeighborsTransformer(n_neighbors=self.n_samples).fit(
                X[:, 1:]
            )
            neighbors = nn.kneighbors(
                current[:, 1:], return_distance=False
            ).reshape(-1)
            samples.append(X[neighbors])
        return np.concatenate(samples, axis=0, dtype=float)

    def _sample_sources(self, coords: np.ndarray) -> np.ndarray:
        """Samples `n_samples` per coordinate"""
        coords = np.atleast_2d(coords)
        if self.bind_to_existing:
            coords = self._knn_sampling(coords)
        else:
            coords = np.repeat(coords, repeats=self.n_samples, axis=0)
        return coords

    def _preprocess_source(self, source: np.ndarray) -> np.ndarray:
        """Validates and sample source if necessary"""
        source = np.atleast_2d(source)

        if source.ndim > 2:
            raise ValueError(
                f"Coordinates must be a 2-dim array. Found {source.ndim}"
            )

        if source.shape[1] != len(self._spatial_columns) + 1:
            raise ValueError(
                f"Sources 1-axis length must match {['t'] + self._spatial_columns} length. Found {source.shape[1]}"
            )

        t0 = source[0, 0]
        if np.any(source[:, 0] != t0):
            raise ValueError("All sources must belong to the same time point")

        return self._sample_sources(source)

    @update_fit
    def __call__(self, source: np.ndarray) -> Union[zarr.Array, np.ndarray]:
        """Computes interpolation given the `source` coordinates

        Parameters
        ----------
        source : np.ndarray
            (N, D) array of N points on the `t`, (`z`, OPTIONAL), `y`, `x` space.

        Returns
        -------
        np.ndarray
            (N, D + 1) first column is the TrackID of each source
        """
        source = self._preprocess_source(source)
        t0 = source[0, 0]

        pos = np.asarray(source[:, 1:])
        shape = pos.shape

        _noise = self._get_noise_function(shape)

        paths = [self._as_track(t0, pos)]
        for t in tqdm(self.time_iter(t0=int(round(t0))), "Computing paths"):
            valid = self._valid_rows(pos)
            X = (pos + _noise())[valid]
            if len(X) == 0:
                break
            pos[valid] = self._models[t].predict(X)
            paths.append(self._as_track(t + self.step, pos))

        paths = np.concatenate(paths, axis=0)
        paths = paths[np.lexsort((paths[:, 1], paths[:, 0]))]

        return self._compute_heatmap(paths) if self.heatmap else paths
