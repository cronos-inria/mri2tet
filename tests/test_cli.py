from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mri2tet import cli


def test_default_output_path_removes_nifti_suffix():
    assert cli._default_output_path(Path("segmentation.nii.gz")) == Path(
        "segmentation.vtu"
    )
    assert cli._default_output_path(Path("segmentation.nii")) == Path(
        "segmentation.vtu"
    )
    assert cli._default_output_path(Path("segmentation.mgz")) == Path(
        "segmentation.vtu"
    )
    assert cli._default_output_path(Path("segmentation.mgh")) == Path(
        "segmentation.vtu"
    )


def test_main_loads_image_and_writes_labelled_vtu(monkeypatch, capsys):
    image = object()
    mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.uint32),
        labels=np.array([7], dtype=np.int32),
    )
    loaded = []
    meshed = []
    written = []
    monkeypatch.setattr(cli.nib, "load", lambda path: loaded.append(path) or image)
    monkeypatch.setattr(
        cli,
        "tetrahedralize",
        lambda *args, **kwargs: meshed.append((args, kwargs)) or mesh,
    )
    monkeypatch.setattr(
        cli.meshio,
        "write",
        lambda path, data, file_format: written.append((path, data, file_format)),
    )

    assert cli.main(
        [
            "segmentation.nii.gz",
            "--tetrahedra-size",
            "0.5",
            "--distance",
            "1.0",
            "--smoothing-sigma",
            "0.8",
        ]
    ) == 0

    assert loaded == ["segmentation.nii.gz"]
    assert meshed == [
        (
            (image,),
            {"tetrahedra_size": 0.5, "distance": 1.0, "smoothing_sigma": 0.8},
        )
    ]
    assert len(written) == 1
    output_path, written_mesh, file_format = written[0]
    assert output_path == Path("segmentation.vtu")
    assert file_format == "vtu"
    assert written_mesh.cells[0].type == "tetra"
    np.testing.assert_array_equal(written_mesh.cell_data["label"][0], [7])


def test_main_accepts_explicit_output_path(monkeypatch):
    mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.uint32),
        labels=np.array([1], dtype=np.int32),
    )
    monkeypatch.setattr(cli.nib, "load", lambda path: object())
    monkeypatch.setattr(cli, "tetrahedralize", lambda *args, **kwargs: mesh)
    written = []
    monkeypatch.setattr(
        cli.meshio,
        "write",
        lambda path, data, file_format: written.append(path),
    )

    cli.main(["input.nii", "--output", "out/custom.vtu"])

    assert written == [Path("out/custom.vtu")]


@pytest.mark.parametrize(
    ("option", "labels", "expected"),
    [
        ("--exclude-labels", [2, 3], [0, 1, 0, 0, 4]),
        ("--include-labels", [2, 3], [0, 0, 2, 3, 0]),
    ],
)
def test_main_filters_input_labels(monkeypatch, option, labels, expected):
    image = cli.nib.Nifti1Image(
        np.array([[[0, 1, 2, 3, 4]]], dtype=np.int16), np.eye(4)
    )
    mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.uint32),
        labels=np.array([1], dtype=np.int32),
    )
    captured = []
    monkeypatch.setattr(cli.nib, "load", lambda path: image)
    monkeypatch.setattr(
        cli,
        "tetrahedralize",
        lambda filtered_image, **kwargs: (
            captured.append(np.asanyarray(filtered_image.dataobj).copy()) or mesh
        ),
    )
    monkeypatch.setattr(cli.meshio, "write", lambda *args, **kwargs: None)

    cli.main(["input.nii", option, *map(str, labels)])

    np.testing.assert_array_equal(captured[0].ravel(), expected)


def test_label_selection_options_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        cli._parse_args(
            ["input.nii", "--exclude-labels", "1", "--include-labels", "2"]
        )
    assert error.value.code == 2


def test_main_rejects_non_vtu_output(capsys):
    with pytest.raises(SystemExit) as error:
        cli._parse_args(["input.nii.gz", "--output", "mesh.mesh"])
    assert error.value.code == 2
    assert "output path must have a .vtu suffix" in capsys.readouterr().err