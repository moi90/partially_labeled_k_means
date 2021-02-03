import numpy as np
import pytest
import scipy.sparse
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_blobs

from constrained_kmeans import ConstrainedKMeans, rejection_matrix


def todense(x):
    if scipy.sparse.issparse(x):
        return x.todense()
    return x


def test_rejection_matrix():
    y = [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]
    y_rejected = [{0}, {0}, {0}, set(), set(), set(), set(), set(), set(), set(), set()]

    r_mat, labels = rejection_matrix(y, y_rejected)

    assert labels == [0, 1]

    assert_array_equal(
        todense(r_mat),
        [
            [True, False],
            [True, False],
            [True, False],
            [False, True,],
            [False, True],
            [False, True],
            [False, True],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ],
    )


def test_rejection_matrix_reject_all():
    y = [-1, -1, -1, 0, 0, 0, 0]
    y_rejected = [{0}, {0}, {0}, set(), set(), set(), set()]

    with pytest.raises(RuntimeError, match="All classes were rejected"):
        rejection_matrix(y, y_rejected)


def test_rejection_matrix_too_few_classes():
    y = [1, 2, 3, 4, 5, 6, 7]
    y_rejected = [set() for _ in range(len(y))]

    with pytest.raises(ValueError):
        rejection_matrix(y, y_rejected, n_classes=2)


def test_constrained_kmeans_relocated_clusters():
    # check that empty clusters are relocated as expected
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])

    y = [-1, -1, -1, -1]
    y_rejected = [set() for _ in range(len(y))]

    # second center too far from others points will be empty at first iter
    init_centers = np.array([[0.4, 0.4], [3, 3]])

    r_mat, labels = rejection_matrix(y, y_rejected, n_classes=2)

    # No given labels
    assert labels == []

    kmeans = ConstrainedKMeans(n_init=1, init=init_centers, verbose=True)
    kmeans.fit(X, r_mat)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.25
    expected_centers = [[0.25, 0], [0.75, 1]]
    expected_n_iter = 3
    expected_sample_distances = [0.0625, 0.0625, 0.0625, 0.0625]

    assert_array_equal(kmeans.labels_, expected_labels)
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert_allclose(kmeans.cluster_centers_, expected_centers)
    assert_allclose(kmeans.sample_distances_, expected_sample_distances)
    assert kmeans.n_iter_ == expected_n_iter


def test_constrained_kmeans_label_persistence():
    X, y = make_blobs()

    # Non-continous labels shouldn't be a problem
    y = 10 + 2 * y

    # Reset 80% labels to -1
    reset_indices = np.random.choice(y.shape[0], round(y.shape[0] * 0.8))
    y[reset_indices] = -1

    r_mat, labels = rejection_matrix(y)

    labels = np.array(labels)

    kmeans = ConstrainedKMeans(n_init=1, verbose=True)
    kmeans.fit(X, r_mat)

    mask_known = y > -1

    assert_array_equal(labels[kmeans.labels_[mask_known]], y[mask_known])
