import operator
from typing import List, Optional, Union

import numpy as np
import scipy.sparse
import scipy.sparse as sp
import scipy.spatial.distance
import sklearn.metrics
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster._kmeans import _init_centroids
from sklearn.utils import check_random_state
from sklearn.utils.extmath import squared_norm


def rejection_matrix(
    y, y_rejected=None, n_classes=None, dense=None, classes=None, offset=None
):
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

    assert len(y) == len(y_rejected)

    if dense is None:
        # For many specified labels, y won't be sparse anyway
        dense = (y > -1).mean() > 0.5

    y = np.asanyarray(y)

    # See which classes are present in y (excluding -1)
    present_classes = np.unique(y)
    assert np.isfinite(present_classes).all()
    present_classes = present_classes[present_classes > -1]

    if classes is None:
        classes = np.sort(present_classes)

        if n_classes is None:
            n_classes = classes.shape[0]
        else:
            if n_classes < classes.shape[0]:
                raise ValueError(
                    f"n_classes is smaller than number of labels ({n_classes} < {classes.shape[0]})"
                )

        if offset is None:
            offset = np.max(classes) + 1 if classes.shape[0] else 0

        if classes.shape[0]:
            assert offset > np.max(classes)

        n_additional = n_classes - classes.shape[0]
        classes = np.concatenate((classes, offset + np.arange(n_additional)))
    else:
        assert set(present_classes) <= set(classes)

    label_to_idx = {x: i for i, x in enumerate(classes)}

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

    return r_mat, classes


def _set_rejected_inf(
    distances,
    rmat: Union[np.ndarray, scipy.sparse.spmatrix],
    row_slice: Optional[slice] = None,
):
    if not scipy.sparse.issparse(rmat):
        if row_slice is not None:
            rmat = rmat[row_slice]

        # Assignments that are rejected are set to inf
        distances[rmat] = np.inf
        return

    ii, jj = rmat.nonzero()

    if row_slice is not None:
        # Do not change ii inplace
        ii = ii - row_slice.start
        mask = (ii >= 0) & (ii < row_slice.stop - row_slice.start)
        ii, jj = ii[mask], jj[mask]

    distances[ii, jj] = np.inf
    return


def _labels_inertia(X, centers, r_mat, *, verbose, reassign_empty, working_memory=None):
    """
    Calculate labels, sample_distances, and inertia for given samples and centers, considering rejections.
    """

    def reduce_chunk(chunk: np.ndarray, start: int):
        # Block rejected assignments
        _set_rejected_inf(chunk, r_mat, slice(start, start + chunk.shape[0]))

        assert (
            np.isfinite(chunk).any(axis=1).all()
        ), "Some samples do not have a feasible label"

        # Select best label
        labels = chunk.argmin(axis=1)

        sample_distances = np.take_along_axis(
            chunk, np.expand_dims(labels, axis=1), axis=1
        ).reshape(-1)

        return labels, sample_distances

    labels, sample_distances = zip(
        *sklearn.metrics.pairwise_distances_chunked(
            X,
            centers,
            reduce_func=reduce_chunk,
            metric="sqeuclidean",
            n_jobs=1,
            working_memory=working_memory,
        )
    )

    labels, sample_distances = np.concatenate(labels), np.concatenate(sample_distances)

    if reassign_empty:
        _reassign_empty_labels(labels, sample_distances, r_mat, verbose=verbose)

    inertia = np.sum(sample_distances)

    return labels, sample_distances, inertia


def _relocatable_samples(
    rmat: Union[np.ndarray, scipy.sparse.spmatrix], relocatable: np.ndarray, label: int
):
    """
    Sample indices that are relocatable to label.

    Relocatable in the first place and not rejected for label.

    Returns:
        Array of relocatable sample indices.
    """

    if not scipy.sparse.issparse(rmat):
        # Samples that are not rejected for label AND relocatable
        return (~rmat[:, label] & relocatable).nonzero()[0]

    ii, jj = rmat.nonzero()
    infeasible_samples = ii[jj == label]
    feasible_samples = np.ones(rmat.shape[0], dtype=bool)
    feasible_samples[infeasible_samples] = False
    feasible_samples &= relocatable

    return feasible_samples.nonzero()[0]


def _reassign_empty_labels(labels, sample_distances, rmat, *, verbose):
    """
    Relocate empty centers to a feasible sample that is far from its own center.

    Modifies labels and sample_distances in place.
    """

    n_centers = rmat.shape[1]
    label_count = np.bincount(labels, minlength=n_centers)
    empty_labels = (label_count == 0).nonzero()[0]
    n_empty = empty_labels.shape[0]

    if n_empty == 0:
        return

    if verbose > 0:
        print(f"Relocating {n_empty} empty centers...")

    # Exclude small clusters as donors
    relocatable = np.ones(labels.shape[0], dtype=bool)
    median_label_count = np.median(label_count)
    for l in range(n_centers):
        if label_count[l] < median_label_count:
            relocatable[labels == l] = False

    for l in empty_labels:
        # Select feasible samples that belong to large enough clusters and were not rejected for l
        feasible_samples = _relocatable_samples(rmat, relocatable, l)

        if not len(feasible_samples):
            raise ValueError(f"No feasible samples for label {l}")

        # From feasible samples, select the one with the largest sample_distances (distance to its center)
        best_idx = sample_distances[feasible_samples].argmax()
        best_sample = feasible_samples[best_idx]

        # Label selected sample as l
        labels[best_sample] = l
        # Update sample distance (which is now 0)
        sample_distances[best_sample] = 0


def _constrained_kmeans_single(
    X, r_mat, *, centers_init, max_iter, verbose, tol=0.1, working_memory=None
):
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
            X,
            centers,
            r_mat,
            verbose=verbose,
            reassign_empty=True,
            working_memory=working_memory,
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


def _r_mat_to_y(r_mat: np.ndarray):
    if sp.issparse(r_mat):
        r_mat = r_mat.toarray()

    # Restore y from r_mat
    y = r_mat.argmin(axis=1)
    y[r_mat.sum(axis=1) < r_mat.shape[1] - 1] = -1

    return y


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
    from sklearn.utils.extmath import row_norms

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


def _init_and_run(
    X: np.ndarray,
    y: np.ndarray,
    init,
    init_size,
    r_mat: np.ndarray,
    max_iter,
    verbose,
    tol,
    init_advanced,
    working_memory,
):
    n_clusters = r_mat.shape[1]

    if init_advanced:
        print("Advanced initialization.")

        centers = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)

        # Decode rejection matrix
        y_enc = _r_mat_to_y(r_mat)

        # Initialize centers for determined clusters
        i = 0
        for i in range(n_clusters):
            mask = y_enc == i
            if mask.sum() == 0:
                break
            centers[i] = X[mask].mean(axis=0)

        if verbose:
            print(
                f"Initializing {i} labeled and {n_clusters - i} unlabeled cluster centers..."
            )
            print("init_size:", init_size)

        if n_clusters - i > 0:
            # Initialize cententers for remaining free clusters from free objects (y==-1)
            centers[i:] = _init_centroids(
                X[y == -1], n_clusters - i, init=init, init_size=init_size
            )

    else:
        centers = _init_centroids(X, n_clusters, init=init, init_size=init_size)

    assert centers.shape[0] == r_mat.shape[1]

    (labels, sample_distances, inertia, centers, n_iter_,) = _constrained_kmeans_single(
        X,
        r_mat,
        centers_init=centers,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        working_memory=working_memory,
    )

    return (
        labels,
        sample_distances,
        inertia,
        centers,
        n_iter_,
    )


class ConstrainedKMeans(ClusterMixin, BaseEstimator):
    """

    Args:
        offset (int, optional): Offset for new labels.
        init_size (int, optional): Number of samples to randomly sample for speeding up the
            initialization.
        working_memory (int, optional): Working memory for distance calculations.
            See :py:fun:`sklearn.metrics.pairwise_distances_chunked`.
    """

    def __init__(
        self,
        n_clusters,
        init="k-means++",
        n_init=10,
        init_size=None,
        max_iter=300,
        verbose=0,
        tol=1e-4,
        n_jobs=None,
        init_advanced=False,
        offset=None,
        working_memory=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.init_size = init_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init_advanced = init_advanced
        self.offset = offset
        self.working_memory = working_memory

    def fit(self, X: np.ndarray, y: np.ndarray, y_rejected=None):
        """

        Compute k-means clustering.

        Arguments:
            X (array of shape (n_samples, n_features)): Training instances to cluster.
            y (array of shape (n_samples,)): Labels for training instances. -1 for unknown label.
            y_rejected (list of set, optional): Rejected labels for the training instances.

        Returns:
            self: Fitted estimator.
        """

        r_mat, self.classes_ = rejection_matrix(
            y, y_rejected, self.n_clusters, offset=self.offset
        )

        if scipy.sparse.issparse(r_mat):
            # COO is most efficient for nonzero in _labels_inertia
            r_mat = r_mat.tocoo()

        # Run clustering n_init times
        values: List = Parallel(n_jobs=self.n_jobs, verbose=self.verbose - 1)(
            delayed(_init_and_run)(
                X,
                y,
                self.init,
                self.init_size,
                r_mat,
                self.max_iter,
                self.verbose,
                self.tol,
                self.init_advanced,
                self.working_memory,
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

    def fit_predict(self, X: np.ndarray, y: np.ndarray, y_rejected=None):
        self.fit(X, y, y_rejected)
        return self.classes_[self.labels_], self.sample_distances_

    def predict(self, X: np.ndarray, y: np.ndarray, y_rejected=None):
        assert (
            X.shape[0] == y.shape[0]
        ), f"Shapes of X and y do not match: {X.shape} vs. {y.shape}"

        r_mat, _ = rejection_matrix(
            y, y_rejected, self.n_clusters, classes=self.classes_
        )

        if scipy.sparse.issparse(r_mat):
            # COO is most efficient for nonzero in _labels_inertia
            r_mat = r_mat.tocoo()

        # Do not reassign empty clusters because we're just predicting
        labels, sample_distances, _ = _labels_inertia(
            X,
            self.cluster_centers_,
            r_mat,
            verbose=self.verbose,
            reassign_empty=False,
            working_memory=self.working_memory,
        )

        return self.classes_[labels], sample_distances
