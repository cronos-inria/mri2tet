"""Type declarations for the private CGAL Mesh_3 extension."""

from typing import TypeAlias

import numpy as np

LabelVolume: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.int32]
]
Labels: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int32]]
Vertices: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Tetrahedra: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
Affine: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Bounds: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def create(
    labels: LabelVolume,
    world_to_voxel: Affine,
    affine: Affine,
    bounds: Bounds,
    tetrahedra_size: float,
    distance: float,
    smoothing_sigma: float,
) -> tuple[Vertices, Tetrahedra, Labels]:
    """Create a labelled tetrahedral mesh from a voxel label array.

    :param labels: C-contiguous ``int32`` array with shape ``(i, j, k)``.
    :param world_to_voxel: C-contiguous inverse NIfTI affine with shape
        ``(4, 4)``.
    :param affine: C-contiguous forward image affine with shape ``(4, 4)``.
    :param bounds: C-contiguous world-coordinate bounds with shape ``(2, 3)``,
        ordered as lower and upper corners.
    :param tetrahedra_size: Maximum target cell circumradius in world units.
    :param distance: Maximum target facet approximation distance in world
        units.
    :param smoothing_sigma: Gaussian label-weights standard deviation in
        world units.
    :returns: Vertex coordinates, tetrahedron vertex indices, and cell labels.
    """
    ...
