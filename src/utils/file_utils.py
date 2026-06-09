from pathlib import Path
from zipfile import ZipFile
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


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
