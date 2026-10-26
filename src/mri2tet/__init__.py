"""Tetrahedral meshing of labelled MRI volumes.

The :func:`tetrahedralize` function generates a :class:`TetrahedralMesh` from
a three-dimensional NIfTI label image.
"""

from .mesh import TetrahedralMesh, tetrahedralize

__all__ = ["TetrahedralMesh", "tetrahedralize"]
