# mri2tet

`mri2tet` generates tetrahedral meshes from labelled 3D nibabel images using CGAL Mesh_3 and exposes results as NumPy arrays.

## Development

The reproducible development build is currently supported on 64-bit Linux via
Pixi. Install Pixi, then run `pixi install` to create the environment. The
environment includes the C++ compiler, CGAL's native dependencies, CMake,
Ninja, and Python build dependencies. CGAL is checked out from its source
repository at the ref set by `CGAL_REF` in `pixi.toml`. Set
that value to a full commit SHA to pin an exact revision. The `build` task
fetches that source and compiles `mri2tet` against it. CGAL is header-only, so
there is no separate CGAL library compilation step. Run `pixi run build` to
compile and install the package, or `pixi run test` to rebuild it and run the
test suite.

To build a source archive, run `pixi run --locked sdist`. To verify that archive
can build a wheel in the same locked environment, run:

```sh
pixi run --locked python -m build --no-isolation --wheel dist/mri2tet-0.1.0.tar.gz
```

Building from source with pip outside Pixi is not self-contained: it requires a
C++17 compiler, CMake, Ninja, pybind11, scikit-build-core, and a CGAL
installation whose CMake package can be found. The committed Pixi lockfile
covers `linux-64`; other platforms and Python versions require their own
validated environment and lockfile.

## Quick start

```python
import nibabel as nib
import mri2tet

mesh = mri2tet.tetrahedralize(nib.load("labels.nii.gz"), tetrahedra_size=2.0)
vertices = mesh.vertices
cells = mesh.tetrahedra
labels = mesh.labels
```

The `mri2tet` command creates a labelled VTU mesh directly from a NIfTI or
FreeSurfer MGH/MGZ image:

```sh
mri2tet segmentation.nii.gz --tetrahedra-size 0.5 --distance 1.0 --smoothing-sigma 0.8
```

By default, this writes `segmentation.vtu` beside the input image. Use `-o` to
choose another `.vtu` output path. Cell labels are stored as `label` data.
Use `--exclude-labels LABEL [LABEL ...]` to set selected input labels to zero,
or `--include-labels LABEL [LABEL ...]` to set every label not selected to
zero. These options are mutually exclusive. For example, to mesh only labels
1 and 3, pass `--include-labels 1 3`.
The equivalent Pixi task forwards arguments to the CLI:

```sh
pixi run --locked mri2tet segmentation.nii.gz --tetrahedra-size 0.5 --distance 1.0 --smoothing-sigma 0.8
```

## Python API

`mri2tet.tetrahedralize(image, tetrahedra_size=1.0, distance=1.0, smoothing_sigma=1.0)` accepts a
three-dimensional nibabel spatial image, including NIfTI and FreeSurfer
MGH/MGZ images. Its data must be finite, integer-valued labels representable
as `int32`, with at least one nonzero voxel. Zero denotes background. Each
distinct nonzero integer label defines a subdomain and is retained as the label
for output tetrahedra.

Voxel indices identify voxel centers. Every voxel is treated as a box extending
half a voxel in each direction from its center. The image affine must be finite,
invertible, and have homogeneous bottom row `[0, 0, 0, 1]`. Translation,
rotation, reflection, anisotropic scaling, and shear are supported. Vertex
coordinates and both meshing criteria use the image's world-coordinate units.
`tetrahedra_size` is a positive finite upper bound on tetrahedron circumradius,
`distance` is a positive finite maximum facet approximation distance, and
`smoothing_sigma` is a positive finite Gaussian standard deviation used by
the label-weight mechanism, in world-coordinate units. Increasing it smooths
the represented label boundaries more strongly, but can erase thin structures.
The smoother computes weights for all labels, including background, as CGAL's
weight generator does. It uses the lengths of the voxel-axis vectors as image
spacing. For sheared affines this is an approximation to isotropic smoothing
in world space.

The result is a `TetrahedralMesh` with `vertices` shaped `(n_vertices, 3)` and
dtype `float64`, `tetrahedra` shaped `(n_tetrahedra, 4)` and dtype `uint32`, and
`labels` shaped `(n_tetrahedra,)` and dtype `int32`. Vertex coordinates are in
world space. Tetrahedron entries index rows of `vertices`, and each entry in
`labels` corresponds to the tetrahedron on the same row. The dataclass is
frozen, but its NumPy arrays are mutable.

## Licensing

This project is licensed under the GNU General Public License, version 3 or any
later version. It uses CGAL's Mesh_3 and 3D Triangulations packages, which are
licensed under the GPL. See [LICENSE](LICENSE) for the license identifier and
links to the full terms, and review the CGAL [license information](https://www.cgal.org/license.html)
before redistribution.
