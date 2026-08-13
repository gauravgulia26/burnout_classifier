# 🔥 Burnout AI Risk Prediction

An end-to-end Machine Learning project focused on predicting **student burnout risk levels** using academic performance, AI usage patterns, study habits, and behavioral indicators.

The project is being built with a strong emphasis on:

✅ Modular Architecture
✅ Scalability
✅ Reproducibility
✅ MLOps Readiness
✅ Software Engineering Best Practices

The goal is not just to train a model, but to build a complete machine learning system that can evolve into a production-ready solution with tools such as DVC, MLflow, automated pipelines, and experiment tracking.

## Run the project

Create an environment with Python 3.12, install dependencies, then train the full pipeline:

```bash
pip install -r requirements.txt
python main.py train
```

This runs ingestion, schema validation, profiling, feature engineering, transformation, model selection, tuning, and registration. The tuning stage grid-searches the experiment-winning model and directly trains the refitted best estimator on the full training split, writing the inference-ready bundle to `models/burnout_classifier.joblib`.

Before running model selection, start an MLflow server and set its URL in `configs/experiment_params.yaml` under `experiment.tracking.tracking_uri`:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

The experiment stage runs each enabled model once with its estimator default parameters using cross-validation, logs the defaults and metrics to MLflow, and writes only the winning selection to `artifacts/experiment_artifact/experiment.json`. The tuning stage consumes that selection, grid-searches the winning model, and saves the refitted best estimator (with the transformation preprocessor) as the inference-ready bundle — so no separate training stage is needed.

The model registry stage (`src/pipeline/stage_09_model_registry.py`) loads the preprocessor pipeline produced by the data-transformation component, appends the tuned best model as its final classifier step, and registers that full inference pipeline into the MLflow Model Registry along with its signature, a sample input dataset, and the preprocessor artifacts. Pass the MLflow server URI explicitly, or let it default to `configs/registry_params.yaml`:

```bash
python src/pipeline/stage_09_model_registry.py --tracking-uri http://127.0.0.1:5000
```

The registry hand-off is written to `artifacts/model_registry_artifact/model_registry.json` and contains the registered model URI (`models:/<name>/<version>`), run id, model/preprocessor/sample artifact URIs, and the test metrics — everything an inference service needs to load the model (`mlflow.pyfunc.load_model("models:/<name>/<version>")`).

To predict from a CSV containing the raw input columns (the target column is optional):

```bash
python src/modeling/predict.py input.csv --predictions-path predictions.csv
```

Run automated checks with `python -m pytest -q`. DVC users can reproduce individual stages with `dvc repro`.

## Inference API

The separate FastAPI backend lives in `src/backend`. It validates raw student records with Pydantic, generates the project features, and loads the registered MLflow model from the tracking server.

Set the registry location (the defaults point to the registered `champion` model) and start the API:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/grvgulia007/burnout_classifier.mlflow
export MLFLOW_MODEL_URI='models:/burnout_classifier@champion'
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

Send one or more records to `POST /api/v1/predict`; interactive API documentation is at `/docs`.

## Streamlit frontend

The separate Streamlit frontend in `src/frontend` sends assessments only through the FastAPI backend. Start the backend first, then run:

```bash
export BACKEND_API_URL=http://127.0.0.1:8000/api/v1
streamlit run src/frontend/app.py
```

The frontend caches its reusable HTTP client and short-lived health checks; predictions are never cached.

## Docker deployment

One container runs the FastAPI model service internally and exposes the Streamlit interface on port `8501`. The root `main.py` starts Streamlit only after FastAPI confirms that the registered MLflow model is loaded; the frontend also displays a waiting screen if its API is unavailable. `serve` is the default command, while `train` runs the original complete training pipeline.

```bash
docker build -t burnout-classifier .
docker run --rm -p 8501:8501 \
  -e MLFLOW_TRACKING_URI="http://your-mlflow-server:5000" \
  -e MLFLOW_MODEL_URI="models:/burnout_classifier@champion" \
  burnout-classifier
```

Open [http://localhost:8501](http://localhost:8501). If the tracking server requires credentials, provide its standard MLflow authentication environment variables (for example `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`) to `docker run` as well.

Alternatively, configure the same variables in your shell and run `docker compose up --build`.

## GitHub Actions CI/CD

The workflow in `.github/workflows/ci-cd.yml` runs linting and tests on every push. On pushes to `main` (or a manual dispatch), it restores DVC data, force-reproduces the pipeline through `model_registry`, and publishes both `latest` and commit-SHA Docker Hub image tags. The deployed container loads the `champion` MLflow model alias at startup.

Configure these repository secrets before enabling deployments:

- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` — Docker Hub account and access token.
- `DAGSHUB_USER_TOKEN` — token permitted to log/register models to the configured Dagshub MLflow server.
- `DVC_REMOTE_URL` — DVC remote URL containing the raw data and pipeline artifacts.
- `DVC_REMOTE_USERNAME` and `DVC_REMOTE_PASSWORD` — required only for a protected DVC HTTP remote.
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` — optional overrides when your MLflow server requires explicit configuration; Dagshub uses `DAGSHUB_USER_TOKEN`.

---

# 🏗️ Project Structure

```text
burnout_ai_risk_prediction/
│
├── artifacts/
├── configs/
├── data/
├── docs/
├── logs/
├── models/
├── notebooks/
├── references/
├── reports/
├── src/
├── tests/
│
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── pyproject.toml
└── main.py
```

---

# 📁 Directory Overview

## 📦 artifacts/

Stores outputs generated by different pipeline stages.

Examples:

* Component artifacts
* Metadata files
* Validation reports
* Feature statistics
* Model outputs

This directory serves as the communication layer between different stages of the pipeline and is designed to be compatible with DVC tracking.

---

## ⚙️ configs/

Contains all project configuration files.

Examples:

* Dataset schema definitions
* Pipeline settings
* Hyperparameters
* Environment configurations

Keeping configuration separate from code improves maintainability and reproducibility.

---

## 📊 data/

Central storage location for datasets throughout their lifecycle.

Typical flow:

```text
Raw Data
    ↓
Interim Data
    ↓
Processed Data
```

This separation helps maintain data lineage and reproducibility.

---

## 📚 docs/

Contains project documentation and documentation site assets.

Can be used with documentation tools such as MkDocs for generating professional project documentation.

---

## 📝 logs/

Stores execution logs generated by pipeline components.

Examples:

* Data Ingestion Logs
* Data Validation Logs
* Training Logs
* Prediction Logs

Useful for monitoring, debugging, and auditing pipeline executions.

---

## 🤖 models/

Stores trained machine learning assets.

Examples:

* Trained models
* Encoders
* Scalers
* Feature transformers
* Model metadata

---

## 📓 notebooks/

Contains Jupyter notebooks used during experimentation.

Typical activities:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Prototyping
* Experimentation

Notebooks are intentionally separated from production code.

---

## 📖 references/

Stores external resources and project references.

Examples:

* Research papers
* Dataset documentation
* Technical notes
* Domain knowledge resources

---

## 📈 reports/

Contains generated reports and visual outputs.

Examples:

* EDA Reports
* Model Evaluation Reports
* Feature Importance Reports
* Visualizations and Charts

---

## 🧠 src/

The core source code of the project.

Contains all implementation related to:

* Pipeline Components
* Configuration Management
* Artifact Definitions
* Validation Logic
* Utility Functions
* Training & Prediction Workflows

The codebase follows modular design principles to encourage maintainability and scalability.

---

## 🧪 tests/

Contains automated tests for validating project functionality.

Examples:

* Unit Tests
* Validator Tests
* Utility Tests
* Component Tests

Helps ensure reliability as the project evolves.

---

# 📄 Root Files

## 🚀 main.py

Primary application entry point.

Responsible for triggering and orchestrating project workflows.

---

## 📦 requirements.txt

Contains all project dependencies required to run the application.

---

## 🏗️ pyproject.toml

Modern Python project configuration file used for dependency management, packaging, and tooling.

---

## ⚡ Makefile

Provides shortcuts for common development tasks.

Examples:

* Running pipelines
* Executing tests
* Formatting code
* Cleaning artifacts

---

## 📜 LICENSE

Defines licensing information for the project.

---

## 📘 README.md

Project overview, architecture explanation, setup instructions, and usage documentation.

---

# 🏛️ Design Principles

This project is intentionally designed around software engineering practices commonly found in production ML systems.

### 🔹 Single Responsibility Principle (SRP)

Each component is responsible for one specific task.

---

### 🔹 Dependency Injection

Dependencies are supplied externally instead of being hardcoded.

---

### 🔹 Loose Coupling

Components communicate through artifacts rather than direct dependencies.

---

### 🔹 Configuration-Driven Development

Project behavior is controlled through configuration files rather than modifying source code.

---

### 🔹 Artifact-Based Communication

Pipeline stages exchange information through versionable artifacts.

---

### 🔹 Modular Pipeline Architecture

Each pipeline stage remains independently maintainable and testable.

---

### 🔹 Reproducible Experimentation

Supports future integration with experiment tracking and version control systems.

---

### 🔹 DVC & MLflow Readiness

The architecture is designed with future MLOps adoption in mind.

---

# 🔄 High-Level Pipeline

```text
📥 Data Ingestion
        ↓
🔍 Data Validation
        ↓
🛠️ Data Transformation
        ↓
🤖 Model Training
        ↓
📊 Model Evaluation
        ↓
🚀 Prediction
```

Each stage produces artifacts that become inputs for downstream stages, creating a reproducible and scalable machine learning workflow.

---

# 🎯 Project Goal

The objective of this project is to move beyond traditional notebook-based machine learning and build a complete, production-oriented ML system capable of:

* Predicting burnout risk levels
* Supporting reproducible experimentation
* Enabling scalable development
* Following software engineering best practices
* Serving as a foundation for future MLOps deployment

> Building a model is important. Building a maintainable ML system is the real challenge.
