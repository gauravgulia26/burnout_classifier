"""Batch inference command for a trained burnout-risk model."""

from pathlib import Path

import joblib
import pandas as pd
import typer

from src.configs.paths import MODEL_PATH
from src.features.make_features import FeatureEngineer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    features_path: Path = typer.Argument(..., exists=True, readable=True),
    predictions_path: Path = typer.Option(Path("data/processed/predictions.csv")),
    model_path: Path = typer.Option(MODEL_PATH, exists=True, readable=True),
) -> None:
    """Predict Low, Medium, or High burnout risk for rows in a CSV file."""
    raw = pd.read_csv(features_path)
    engineer = FeatureEngineer(features_path, predictions_path, predictions_path)
    features = engineer.engineer_dataframe(raw)
    bundle = joblib.load(model_path)
    transformed = bundle["preprocessor"].transform(features)
    encoded = bundle["model"].predict(transformed)
    labels = {int(key): value for key, value in bundle["label_map"].items()}
    output = raw.copy()
    output["Burnout_Risk_Prediction"] = [labels[int(value)] for value in encoded]
    if hasattr(bundle["model"], "predict_proba"):
        output["Prediction_Confidence"] = bundle["model"].predict_proba(transformed).max(axis=1)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False)
    typer.echo(f"Saved {len(output)} predictions to {predictions_path}")


if __name__ == "__main__":
    app()
