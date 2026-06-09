from pathlib import Path


def validate_path(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Non-Existent Path: {file_path}")
