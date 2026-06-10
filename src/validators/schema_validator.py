import pandas as pd
from pathlib import Path


def validate_schema(
    df_path: Path,
    schema: dict,
) -> None:
    """
    Validate dataframe against schema definition.
    """
    df = pd.read_csv(df_path)
    expected_columns = schema["columns"]

    actual_columns = set(df.columns)

    required_columns = set(expected_columns.keys())

    missing_columns = required_columns - actual_columns

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    for column_name, metadata in expected_columns.items():

        expected_type = metadata["type"]

        series = df[column_name]

        if expected_type == "numerical" and not pd.api.types.is_numeric_dtype(series):
            raise TypeError(f"{column_name} should be numerical.")

        if expected_type == "categorical" and not (
            pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        ):
            raise TypeError(f"{column_name} should be categorical.")

        if expected_type == "boolean" and not pd.api.types.is_bool_dtype(series):
            raise TypeError(f"{column_name} should be boolean.")
