"""Generate tetrahedral meshes from labelled nibabel spatial images.

The public API converts a three-dimensional :class:`nibabel.spatialimages.SpatialImage`
to a :class:`TetrahedralMesh` using CGAL Mesh_3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from nibabel.affines import apply_affine
from nibabel.spatialimages import SpatialImage
import numpy as np

from . import _mesh

LabelVolume: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.int32]
]
Vertices: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Tetrahedra: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
CellLabels: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int32]]
Affine: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Bounds: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


@dataclass(frozen=True, slots=True)
class TetrahedralMesh:
    """A tetrahedral mesh with an integer label for each cell.

    Vertices are expressed in the input image's world coordinate system.
    Tetrahedron indices refer to rows of :attr:`vertices`, and each value in
    :attr:`labels` corresponds to the tetrahedron in the same row of
    :attr:`tetrahedra`. The arrays themselves remain mutable despite the
    frozen dataclass.

    .. attribute:: vertices

         ``float64`` array with shape ``(n_vertices, 3)`` containing vertex
       coordinates.

    .. attribute:: tetrahedra

       ``uint32`` array with shape ``(n_tetrahedra, 4)`` containing vertex
       indices for each tetrahedron.

    .. attribute:: labels

       ``int32`` array with shape ``(n_tetrahedra,)`` containing the input
       image label associated with each tetrahedron.
    """

    vertices: Vertices
    tetrahedra: Tetrahedra
    labels: CellLabels


def _positive_finite(value: float, name: str) -> float:
    """Convert a scalar to ``float`` and require a finite positive value.

    :param value: Value to validate.
    :param name: Parameter name used in validation errors.
    :returns: The converted value.
    :raises TypeError: If ``value`` cannot be converted to a real scalar.
    :raises ValueError: If the converted value is not finite and positive.
    """
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar.") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return converted


def tetrahedralize(
    image: SpatialImage,
    tetrahedra_size: float = 1.0,
    distance: float = 1.0,
    smoothing_sigma: float = 1.0,
) -> TetrahedralMesh:
    """Generate a tetrahedral mesh from a labelled 3-D nibabel image.

    Label value zero denotes background. Every distinct nonzero integer label
    defines a separate CGAL subdomain and is retained on the output cells.
    The image affine maps voxel coordinates to world coordinates. It may
    include translation, rotation, reflection, anisotropic scaling, and shear,
    but must be finite and invertible, with homogeneous bottom row
    ``[0, 0, 0, 1]``.

    Mesh vertices and the meshing criteria are expressed in world coordinates.
    Each voxel is treated as a box extending half a voxel from its center.

    :param image: Three-dimensional nibabel spatial image containing finite,
        integer-valued labels that fit in ``int32``. At least one voxel must
        have a nonzero label.
    :param tetrahedra_size: Upper bound on the tetrahedron circumradius, in
        world-coordinate units. Must be finite and positive.
    :param distance: Maximum facet approximation distance, in world-coordinate
        units. Must be finite and positive.
    :param smoothing_sigma: Standard deviation for CGAL's Gaussian label
        weights, in world-coordinate units. Larger values smooth boundaries
        more strongly but may remove thin structures. Must be finite and
        positive.
    :returns: A :class:`TetrahedralMesh` containing world-coordinate vertices,
        tetrahedron connectivity, and one label per tetrahedron.
    :raises TypeError: If the image data are not numeric.
    :raises ValueError: If the image is not 3-D, contains invalid labels, has
        an empty labelled domain, or has a non-finite or non-invertible affine.
    """
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError("image must be three-dimensional.")
    if any(size == 0 for size in data.shape):
        raise ValueError("image dimensions must be non-empty.")
    try:
        finite = np.isfinite(data)
    except TypeError as error:
        raise TypeError("image data must be numeric.") from error
    if not np.all(finite):
        raise ValueError("image data must contain only finite values.")
    if not np.all(data == np.rint(data)):
        raise ValueError("image labels must be integer-valued.")
    if np.any(data < np.iinfo(np.int32).min) or np.any(data > np.iinfo(np.int32).max):
        raise ValueError("image labels must fit in int32.")

    label_volume = cast(
        LabelVolume, np.ascontiguousarray(data, dtype=np.int32)
    )
    if not np.any(label_volume != 0):
        raise ValueError("image must contain at least one nonzero label.")

    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("image affine must be a finite 4 by 4 matrix.")
    if not np.array_equal(affine[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(
            "image affine must have homogeneous bottom row [0, 0, 0, 1]."
        )
    try:
        world_to_voxel = cast(
            Affine,
            np.ascontiguousarray(np.linalg.inv(affine), dtype=np.float64),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("image affine must be invertible.") from error
    if not np.all(np.isfinite(world_to_voxel)):
        raise ValueError("image affine inverse must be finite.")

    voxel_bounds = np.array(
        [
            [i, j, k]
            for i in (-0.5, data.shape[0] - 0.5)
            for j in (-0.5, data.shape[1] - 0.5)
            for k in (-0.5, data.shape[2] - 0.5)
        ],
        dtype=np.float64,
    )
    world_corners = apply_affine(affine, voxel_bounds)
    world_bounds = cast(
        Bounds,
        np.ascontiguousarray(
            np.stack((world_corners.min(axis=0), world_corners.max(axis=0))),
            dtype=np.float64,
        ),
    )
    forward_affine = cast(
        Affine, np.ascontiguousarray(affine, dtype=np.float64)
    )
    if not np.all(np.isfinite(world_bounds)):
        raise ValueError("image affine must produce finite world-space bounds.")

    vertices, tetrahedra, labels = _mesh.create(
        label_volume,
        world_to_voxel,
        forward_affine,
        world_bounds,
        _positive_finite(tetrahedra_size, "tetrahedra_size"),
        _positive_finite(distance, "distance"),
        _positive_finite(smoothing_sigma, "smoothing_sigma"),
    )
    return TetrahedralMesh(
        cast(Vertices, np.ascontiguousarray(vertices, dtype=np.float64)),
        cast(Tetrahedra, np.ascontiguousarray(tetrahedra, dtype=np.uint32)),
        cast(CellLabels, np.ascontiguousarray(labels, dtype=np.int32)),
    )
