import json

import numpy as np
import pandas as pd


class ValidationProfiler:

    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.df = df

    def generate_statistics(
        self,
        output_path: str,
    ) -> None:

        numerical_df = self.df.select_dtypes(include="number")

        statistics = {}

        for column in numerical_df.columns:

            statistics[column] = {
                "mean": float(numerical_df[column].mean()),
                "median": float(numerical_df[column].median()),
                "std": float(numerical_df[column].std()),
                "min": float(numerical_df[column].min()),
                "max": float(numerical_df[column].max()),
                "q25": float(numerical_df[column].quantile(0.25)),
                "q50": float(numerical_df[column].quantile(0.50)),
                "q75": float(numerical_df[column].quantile(0.75)),
                "skewness": float(numerical_df[column].skew()),
            }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                statistics,
                file,
                indent=4,
            )

    def generate_missing_values(
        self,
        output_path: str,
    ) -> None:

        report = {}

        total_rows = len(self.df)

        for column in self.df.columns:

            missing_count = int(self.df[column].isna().sum())

            report[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(
                    (missing_count / total_rows) * 100,
                    2,
                ),
            }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

    def generate_distributions(
        self,
        output_path: str,
    ) -> None:

        report = {}

        numerical_df = self.df.select_dtypes(include="number")

        for column in numerical_df.columns:

            hist, bins = np.histogram(
                numerical_df[column],
                bins=10,
            )

            report[column] = {
                "histogram": hist.tolist(),
                "bins": bins.tolist(),
            }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

    def generate_summary(
        self,
        output_path: str,
    ) -> None:

        summary = {
            "num_rows": int(self.df.shape[0]),
            "num_columns": int(self.df.shape[1]),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "numerical_features": len(self.df.select_dtypes(include="number").columns),
            "categorical_features": len(
                self.df.select_dtypes(
                    include=[
                        "object",
                        "category",
                    ]
                ).columns
            ),
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )
