"""Command-line interface for generating VTU meshes from nibabel images.

The ``mri2tet`` command loads a labelled image, tetrahedralizes it, and writes
the resulting mesh and cell labels to a VTU file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, cast

import meshio
import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage

from .mesh import tetrahedralize


def _default_output_path(input_path: Path) -> Path:
    """Return a sibling VTU path, removing the complete image suffix.

    Recognized compound and single-file image suffixes are removed so that,
    for example, ``scan.nii.gz`` produces ``scan.vtu`` rather than
    ``scan.nii.vtu``. Other paths use :attr:`Path.stem`.

    :param input_path: Input image path.
    :returns: Output path in the same directory with a ``.vtu`` suffix.
    """
    name = input_path.name
    lower_name = name.lower()
    # Match the compound suffix before checking the single-file formats.
    if lower_name.endswith(".nii.gz"):
        stem = name[:-7]
    elif lower_name.endswith((".nii", ".mgz", ".mgh")):
        stem = name[:-4]
    else:
        stem = input_path.stem
    return input_path.with_name(f"{stem}.vtu")


def _filter_labels(
    image: SpatialImage,
    include_labels: Sequence[int] | None,
    exclude_labels: Sequence[int] | None,
) -> SpatialImage:
    """Return an image with labels outside the selection set to zero."""
    if include_labels is None and exclude_labels is None:
        return image

    data = np.asanyarray(image.dataobj).copy()
    if include_labels is not None:
        data[~np.isin(data, include_labels)] = 0
    elif exclude_labels is not None:
        data[np.isin(data, exclude_labels)] = 0
    return cast(
        SpatialImage,
        image.__class__(data, image.affine, image.header.copy()),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments and resolve the output path.

    :param argv: Argument strings to parse, or ``None`` to use
        the process command line.
    :returns: Parsed options, including a resolved ``output`` path.
    :raises SystemExit: If the arguments are invalid or help is requested.
    """
    parser = argparse.ArgumentParser(
        prog="mri2tet",
        description="Create a labelled tetrahedral VTU mesh from an image.",
    )
    parser.add_argument("image", type=Path, help="input labelled image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output VTU path (default: input name with a .vtu suffix)",
    )
    parser.add_argument(
        "--tetrahedra-size",
        type=float,
        default=1.0,
        help="maximum tetrahedron circumradius in world units (default: 1.0)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.0,
        help="maximum facet approximation distance in world units (default: 1.0)",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.0,
        help="Gaussian label-smoothing sigma in world units (default: 1.0)",
    )
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--exclude-labels",
        type=int,
        nargs="+",
        metavar="LABEL",
        help="set these input labels to zero before meshing",
    )
    label_group.add_argument(
        "--include-labels",
        type=int,
        nargs="+",
        metavar="LABEL",
        help="keep only these input labels and set all others to zero",
    )
    args = parser.parse_args(argv)
    args.output = args.output or _default_output_path(args.image)
    if args.output.suffix.lower() != ".vtu":
        parser.error("output path must have a .vtu suffix")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a VTU mesh from a labelled image.

    The input is loaded with nibabel and passed to :func:`tetrahedralize`.
    The resulting vertices, tetrahedron connectivity, and cell labels are
    written as a VTU file.

    :param argv: Argument strings to parse, or ``None`` to use the process
        command line.
    :returns: Zero after the mesh has been written successfully.
    :raises SystemExit: If image loading, tetrahedralization, or file writing
        fails. Argument parsing may also raise this exception for invalid input.
    """
    args = _parse_args(argv)
    try:
        image = cast(SpatialImage, nib.load(str(args.image)))
        image = _filter_labels(
            image, args.include_labels, args.exclude_labels
        )
        result = tetrahedralize(
            image,
            tetrahedra_size=args.tetrahedra_size,
            distance=args.distance,
            smoothing_sigma=args.smoothing_sigma,
        )
        print(len(result.vertices))
        print(len(result.tetrahedra))
        output = meshio.Mesh(
            points=result.vertices,
            cells=[("tetra", result.tetrahedra)],
            cell_data={"label": [result.labels]},
        )
        meshio.write(args.output, output, file_format="vtu")
    except Exception as error:
        # Present library failures in the same concise format as CLI errors.
        raise SystemExit(f"mri2tet: error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
