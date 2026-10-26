from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from nibabel.freesurfer.mghformat import MGHImage

import mri2tet


def test_tetrahedralize_returns_labelled_mesh():
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1:4, 1:3, 1:4] = 1
    data[1:4, 3:4, 1:4] = 2
    affine = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, -4.0],
            [0.0, 0.0, 2.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = nib.Nifti1Image(data, affine)
    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=2.0, distance=0.5)

    assert mesh.vertices.ndim == 2 and mesh.vertices.shape[1] == 3
    assert mesh.vertices.dtype == np.float64
    assert mesh.tetrahedra.ndim == 2 and mesh.tetrahedra.shape[1] == 4
    assert mesh.labels.shape == (len(mesh.tetrahedra),)
    assert set(np.unique(mesh.labels)) == {1, 2}
    assert np.all(mesh.tetrahedra < len(mesh.vertices))
    assert np.all(np.isfinite(mesh.vertices))


def test_tetrahedralize_accepts_mgh_image():
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1:4, 1:4, 1:4] = 4
    image = MGHImage(data, np.eye(4))

    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=2.0, distance=0.5)

    assert set(np.unique(mesh.labels)) == {4}


def test_tetrahedralize_rejects_non_integer_labels():
    image = nib.Nifti1Image(np.full((2, 2, 2), 0.5), np.eye(4))
    with pytest.raises(ValueError, match="integer-valued"):
        mri2tet.tetrahedralize(image)


@pytest.mark.parametrize("data", [np.full((2, 2, 2), np.nan), np.full((2, 2, 2), np.inf)])
def test_tetrahedralize_rejects_non_finite_labels(data):
    image = nib.Nifti1Image(data, np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        mri2tet.tetrahedralize(image)


@pytest.mark.parametrize("label", [np.iinfo(np.int32).max + 1, np.iinfo(np.int32).min - 1])
def test_tetrahedralize_rejects_labels_outside_int32(label):
    image = SimpleNamespace(
        dataobj=np.full((2, 2, 2), label, dtype=np.int64), affine=np.eye(4)
    )
    with pytest.raises(ValueError, match="fit in int32"):
        mri2tet.tetrahedralize(image)


def test_tetrahedralize_rejects_non_3d_image():
    image = nib.Nifti1Image(np.ones((2, 2), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="three-dimensional"):
        mri2tet.tetrahedralize(image)


def test_tetrahedralize_rejects_empty_domain():
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="at least one nonzero label"):
        mri2tet.tetrahedralize(image)


@pytest.mark.parametrize("name", ["tetrahedra_size", "distance", "smoothing_sigma"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_tetrahedralize_rejects_invalid_criteria(name, value):
    image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        mri2tet.tetrahedralize(image, **{name: value})


@pytest.mark.parametrize(
    ("affine", "message"),
    [
        (np.diag([1.0, 1.0, 0.0, 1.0]), "invertible"),
        (np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]]), "homogeneous bottom row"),
    ],
)
def test_tetrahedralize_rejects_unsupported_affines(affine, message):
    image = SimpleNamespace(dataobj=np.ones((2, 2, 2), dtype=np.uint8), affine=affine)
    with pytest.raises(ValueError, match=message):
        mri2tet.tetrahedralize(image)


def test_tetrahedralize_rejects_non_finite_affine():
    affine = np.eye(4)
    affine[0, 0] = np.inf
    image = SimpleNamespace(dataobj=np.ones((2, 2, 2), dtype=np.uint8), affine=affine)
    with pytest.raises(ValueError, match="finite 4 by 4"):
        mri2tet.tetrahedralize(image)


def test_tetrahedralize_supports_sheared_scaled_affine():
    data = np.zeros((5, 5, 5), dtype=np.int16)
    data[1:4, 1:4, 1:4] = 7
    affine = np.array(
        [
            [2.0, 0.5, 0.0, 11.0],
            [0.0, 1.5, 0.3, -5.0],
            [0.0, 0.0, 3.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = nib.Nifti1Image(data, affine)

    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=2.0, distance=0.5)

    assert set(np.unique(mesh.labels)) == {7}
    assert np.all(np.isfinite(mesh.vertices))
    assert np.all(mesh.vertices[:, 0] >= 9.0 - 0.5)
    assert np.all(mesh.vertices[:, 0] <= 21.0 + 0.5)


def test_tetrahedralize_preserves_precision_at_large_world_coordinates():
    data = np.ones((2, 2, 2), dtype=np.uint8)
    affine = np.array(
        [
            [0.1, 0.0, 0.0, 1e8],
            [0.0, 0.1, 0.0, -1e8],
            [0.0, 0.0, 0.1, 1e8],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = nib.Nifti1Image(data, affine)

    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=0.2, distance=0.05)

    assert mesh.vertices.dtype == np.float64
    assert np.ptp(mesh.vertices[:, 0]) > 0.1
    assert np.ptp(mesh.vertices[:, 1]) > 0.1
    assert np.ptp(mesh.vertices[:, 2]) > 0.1


def test_tetrahedralize_preserves_negative_label():
    image = nib.Nifti1Image(-np.ones((3, 3, 3), dtype=np.int16), np.eye(4))

    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=1.0, distance=0.25)

    assert set(np.unique(mesh.labels)) == {-1}


@pytest.mark.xfail(
    reason="CGAL's default initialization may omit disconnected label components",
    strict=False,
)
def test_tetrahedralize_meshes_disconnected_label_regions():
    data = np.zeros((12, 5, 5), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 3
    data[8:11, 1:4, 1:4] = 3
    image = nib.Nifti1Image(data, np.eye(4))

    mesh = mri2tet.tetrahedralize(image, tetrahedra_size=1.0, distance=0.25)

    centroids = mesh.vertices[mesh.tetrahedra].mean(axis=1)
    assert np.any(centroids[:, 0] < 5.0)
    assert np.any(centroids[:, 0] > 6.0)
    assert set(np.unique(mesh.labels)) == {3}


