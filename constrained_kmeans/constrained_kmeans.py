import operator
from typing import List

import numpy as np
import scipy.sparse
import scipy.spatial.distance
import sklearn.metrics
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.extmath import squared_norm, row_norms
from sklearn.utils import check_random_state


def rejection_matrix(y, y_rejected=None, n_classes=None, dense=None):
    """
    Calculate rejection matrix.

    Arguments:
        y: Numeric labels. -1 means no label.
        y_rejected (optional): Rejected labels for each sample.
        n_classes (optional): Total number of labels.
    """

    y = np.asarray(y)

    if y_rejected is None:
        y_rejected = [set() for _ in range(len(y))]

    if dense is None:
        # For many specified labels, y won't be sparse anyway
        dense = (y > -1).mean() > 0.5

    y = np.asanyarray(y)

    labels = np.unique(y)
    labels = sorted(labels[labels > -1])

    if n_classes is None:
        n_classes = len(labels)
    else:
        if n_classes < len(labels):
            raise ValueError(
                f"n_classes is smaller than number of labels ({n_classes} < {len(labels)})"
            )

    assert len(y) == len(y_rejected)

    label_to_idx = {x: i for i, x in enumerate(labels)}

    if dense:
        r_mat = np.zeros((len(y), n_classes), dtype=bool)
    else:
        r_mat = scipy.sparse.lil_matrix((len(y), n_classes), dtype=bool)

    for i, (y_i, y_rejected_i) in enumerate(zip(y, y_rejected)):
        if y_i == -1:
            rejected_jj = [label_to_idx[l] for l in y_rejected_i if l in label_to_idx]
            if len(rejected_jj) == n_classes:
                raise RuntimeError("All classes were rejected")
            r_mat[i, rejected_jj] = True
        else:
            r_mat[i] = True
            r_mat[i, label_to_idx[y_i]] = False

    return r_mat, labels


def _labels_inertia(X, centers, r_mat, *, verbose, reassign_empty):
    # Calculate distances
    distances = sklearn.metrics.pairwise_distances(X, centers, "sqeuclidean")

    # Block rejected assignments
    if scipy.sparse.issparse(r_mat):
        # Sparse requires nonzero()
        distances[r_mat.nonzero()] = np.inf
    else:
        # For dense matrices, it is faster not to use nonzero
        distances[r_mat] = np.inf

    assert (
        np.isfinite(distances).any(axis=1).all()
    ), "Some samples do not have a feasible label"

    labels = distances.argmin(axis=1)

    if reassign_empty:
        labels = _reassign_empty_labels(labels, distances, verbose=verbose)

    # Do not reuse distance calculation from _reassign_empty_labels.
    sample_distances = np.take_along_axis(
        distances, np.expand_dims(labels, axis=1), axis=1
    ).reshape(-1)

    inertia = np.sum(sample_distances)

    return labels, sample_distances, inertia


def _reassign_empty_labels(labels, distances, *, verbose):
    """Relocate empty centers to a feasible sample that is far from its own center."""
    n_centers = distances.shape[1]
    label_count = np.bincount(labels, minlength=n_centers)
    empty_labels = (label_count == 0).nonzero()[0]

    n_empty = empty_labels.shape[0]

    if n_empty == 0:
        return labels

    if verbose > 0:
        print(f"Relocating {n_empty} centers...")

    sample_distances = np.take_along_axis(
        distances, np.expand_dims(labels, axis=1), axis=1
    ).reshape(-1)

    assert np.isfinite(sample_distances).all(), "Some distances are not finite."

    # Exclude small clusters as donators
    median_label_count = np.median(label_count)
    for i in range(n_centers):
        if label_count[i] < median_label_count:
            sample_distances[labels == i] = 0

    for l in empty_labels:
        # Select feasible samples (with finite distance)
        feasible_samples = np.isfinite(distances[:, l]).nonzero()[0]

        if not len(feasible_samples):
            raise ValueError(f"No feasible samples for label {l}")

        # From feasible samples, select the one with the largest sample_distances (distance to its center)
        best_idx = sample_distances[feasible_samples].argmax()
        best_sample = feasible_samples[best_idx]

        labels[best_sample] = l
        # Set to zero to avoid multiple assignment
        sample_distances[best_sample] = 0

    return labels


def _constrained_kmeans_single(X, r_mat, *, centers_init, max_iter, verbose, tol=0.1):
    n_centers = r_mat.shape[1]

    centers = centers_init
    centers_old = np.zeros_like(centers)
    labels = None

    strict_convergence = False

    for iteration in range(max_iter):
        # Save old values
        centers_old[:] = centers
        labels_old = labels

        # Label assignment (E-step)
        labels, sample_distances, inertia = _labels_inertia(
            X, centers, r_mat, verbose=verbose, reassign_empty=True
        )

        if labels_old is not None and np.array_equal(labels, labels_old):
            # Strict convergence
            strict_convergence = True
            if verbose > 0:
                print(
                    f"Converged at iteration {iteration}: strict convergence (inertia: {inertia:.2f})."
                )
            break

        # Center calculation (M-step)
        for i in range(n_centers):
            if (labels == i).sum() == 0:
                # On some occasions, _reassign_empty_labels is not able to heal all empty clusters.
                # Hopefully, it will happen in the next iteration.
                if verbose > 0:
                    print(f"Cluster {i} is empty.")
                continue

            centers[i] = X[labels == i].mean(axis=0)
        center_shift_total = squared_norm(centers_old - centers)

        if center_shift_total < tol:
            if verbose > 0:
                print(
                    f"Converged at iteration {iteration}: center shift "
                    f"{center_shift_total} within tolerance {tol} (inertia: {inertia:.2f})."
                )
            break

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        # Do not relocate empty labels here, because if a cluster is empty now, it simply wasn't meant to be.
        labels, sample_distances, inertia = _labels_inertia(
            X, centers, r_mat, verbose=verbose, reassign_empty=False
        )

    return labels, sample_distances, inertia, centers, iteration + 1


def _init_centroids(
    X, k, init, random_state=None, x_squared_norms=None, init_size=None
):
    """
    Args:
        X: array
        k: number of centroids
    """

    random_state = check_random_state(random_state)

    try:
        from sklearn.cluster._kmeans import _init_centroids as sk_init_centroids
    except ImportError:
        pass
    else:
        return sk_init_centroids(
            X,
            k,
            init,
            random_state=random_state,
            x_squared_norms=x_squared_norms,
            init_size=init_size,
        )

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, init=init)

    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)

    return kmeans._init_centroids(
        X,
        x_squared_norms=x_squared_norms,
        init=init,
        random_state=random_state,
        init_size=init_size,
    )


def _init_and_run(X, n_clusters, init, init_size, r_mat, max_iter, verbose, tol):
    centroids = _init_centroids(X, n_clusters, init=init, init_size=init_size)

    assert centroids.shape[0] == r_mat.shape[1]

    (labels, sample_distances, inertia, centers, n_iter_,) = _constrained_kmeans_single(
        X, r_mat, centers_init=centroids, max_iter=max_iter, verbose=verbose, tol=tol,
    )

    return (
        labels,
        sample_distances,
        inertia,
        centers,
        n_iter_,
    )


class ConstrainedKMeans(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        init="k-means++",
        n_init=10,
        init_size=None,
        max_iter=300,
        verbose=0,
        tol=1e-4,
        n_jobs=None,
    ):
        self.init = init
        self.init_size = init_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, r_mat):
        """
        
        Compute k-means clustering.

        Arguments:
            X (array of shape (n_samples, n_features)): Training instances to cluster.
            r_mat (array of shape (n_samples, n_clusters)): Rejection matrix.
                y[i,j] = True if sample i was rejected for cluster j.

        Returns:
            self: Fitted estimator.
        """

        if scipy.sparse.issparse(r_mat):
            # COO is most efficient for nonzero in _labels_inertia
            r_mat = r_mat.tocoo()

        n_clusters = r_mat.shape[1]

        init_size = self.init_size
        if init_size is None:
            init_size = 10 * n_clusters

        # Run clustering n_init times
        values: List = Parallel(n_jobs=self.n_jobs, verbose=self.verbose - 1)(
            delayed(_init_and_run)(
                X,
                n_clusters,
                self.init,
                init_size,
                r_mat,
                self.max_iter,
                self.verbose,
                self.tol,
            )
            for _repetition in range(self.n_init)
        )

        # Sort by inertia
        values.sort(key=operator.itemgetter(2))

        # Store best run
        (
            self.labels_,
            self.sample_distances_,
            self.inertia_,
            self.cluster_centers_,
            self.n_iter_,
        ) = values[0]

        return self

    def fit_predict(self, X, r_mat):
        self.fit(X, r_mat)
        return self.labels_, self.sample_distances_

    def predict(self, X, r_mat):
        if scipy.sparse.issparse(r_mat):
            # COO is most efficient for nonzero in _labels_inertia
            r_mat = r_mat.tocoo()

        # Do not reassign empty clusters because we're just predicting
        labels, sample_distances, _ = _labels_inertia(
            X, self.cluster_centers_, r_mat, verbose=self.verbose, reassign_empty=False
        )

        return labels, sample_distances
