from pathlib import Path
from zipfile import ZipFile
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
import json
from pydantic import BaseModel
import yaml
from box import Box


def unzip_with_progress(
    zip_file_path: str | Path,
    extract_to: str | Path,
) -> Path:
    """
    Extract a ZIP file with a Rich progress bar.

    Parameters
    ----------
    zip_file_path : str | Path
        Path to zip file.

    extract_to : str | Path
        Destination directory.

    Returns
    -------
    Path
        Extraction directory path.
    """

    zip_file_path = Path(zip_file_path)
    extract_to = Path(extract_to)

    if not zip_file_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_file_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_file_path, "r") as zip_ref:
        members = zip_ref.infolist()

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} files"),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
        ) as progress:

            task = progress.add_task(
                "Extracting",
                total=len(members),
            )

            for member in members:
                zip_ref.extract(member, extract_to)
                progress.advance(task)

    return extract_to


def dump_model_to_json(
    model: BaseModel,
    output_path: str | Path,
) -> str:
    """
    Serialize a Pydantic model to a JSON file.

    Returns:
        Path of the generated artifact.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        model.model_dump_json(indent=4),
        encoding="utf-8",
    )

    return str(output_path)


def load_json_artifact(
    artifact_path: str | Path,
) -> dict:
    """
    Load a JSON artifact as a dictionary.
    """
    artifact_path = Path(artifact_path)

    with open(
        artifact_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_yaml(yaml_path: str | Path) -> Box:
    """
    Load YAML file and return Python-Box.
    """
    yaml_path = Path(yaml_path)

    with open(
        yaml_path,
        "r",
        encoding="utf-8",
    ) as file:
        return Box(yaml.safe_load(file))
