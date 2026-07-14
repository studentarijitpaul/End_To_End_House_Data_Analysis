from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"

app = Flask(__name__)


# These are the numeric columns expected by the housing dataset.
NUMERIC_FIELDS = {
    "BHK": {"label": "Bedrooms (BHK)", "min": 1, "max": 12, "step": 1, "value": 2},
    "Size_in_SqFt": {"label": "Built-up area (sq ft)", "min": 100, "max": 100000, "step": 1, "value": 1000},
    "Price_per_SqFt": {"label": "Market rate per sq ft", "min": 1, "max": 1000000, "step": 1, "value": 10000},
    "Year_Built": {"label": "Year built", "min": 1900, "max": 2100, "step": 1, "value": 2015},
    "Floor_No": {"label": "Floor number", "min": 0, "max": 300, "step": 1, "value": 2},
    "Total_Floors": {"label": "Total floors", "min": 1, "max": 300, "step": 1, "value": 10},
    "Age_of_Property": {"label": "Property age (years)", "min": 0, "max": 200, "step": 1, "value": 10},
    "Nearby_Schools": {"label": "Nearby schools", "min": 0, "max": 100, "step": 1, "value": 3},
    "Nearby_Hospitals": {"label": "Nearby hospitals", "min": 0, "max": 100, "step": 1, "value": 2},
}

CATEGORICAL_FIELDS = {
    "State": "State",
    "City": "City",
    "Locality": "Locality",
    "Property_Type": "Property type",
    "Furnished_Status": "Furnishing",
    "Public_Transport_Accessibility": "Public transport access",
    "Parking_Space": "Parking",
    "Security": "Security",
    "Amenities": "Amenities",
    "Facing": "Facing",
    "Owner_Type": "Listed by",
    "Availability_Status": "Availability",
}

MODEL_DESCRIPTIONS = {
    "linear_regression": ("Linear Regression", "A simple benchmark model that estimates a straight-line relationship."),
    "ridge_regression": ("Ridge Regression", "A regularized linear model designed to stay stable with related features."),
    "lasso_regression": ("Lasso Regression", "A regularized linear model that can reduce the influence of less useful features."),
    "decision_tree": ("Decision Tree", "A rule-based model that learns price patterns from property characteristics."),
    "random_forest": ("Random Forest", "An ensemble of decision trees that usually gives steadier estimates."),
    "gradient_boosting": ("Gradient Boosting", "A sequence of small trees that progressively improves its prediction."),
    "xgboost": ("XGBoost", "A boosted-tree model built for high-performance tabular prediction."),
}


def humanize(value: str) -> str:
    """Turn a saved model filename into a readable model name."""
    return MODEL_DESCRIPTIONS.get(value, (value.replace("_", " ").title(), "Saved regression model."))[0]


def get_model_files() -> list[Path]:
    """Return saved estimators without treating the feature file as a model."""
    if not MODELS_DIR.exists():
        return []
    return sorted(file for file in MODELS_DIR.glob("*.pkl") if file.name != "features.pkl")


@lru_cache(maxsize=1)
def get_feature_names() -> list[str]:
    """Load the exact feature order used while training the models."""
    feature_path = MODELS_DIR / "features.pkl"
    if not feature_path.exists():
        raise FileNotFoundError("Models/features.pkl was not found.")
    return list(joblib.load(feature_path))


@lru_cache(maxsize=8)
def load_model(filename: str) -> Any:
    """Load one selected model and keep it in memory for later requests."""
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Models/{filename} was not found.")
    return joblib.load(model_path)


def get_form_fields(feature_names: list[str]) -> list[dict[str, Any]]:
    """Build the form from the feature columns saved during training."""
    fields: list[dict[str, Any]] = []

    for name, details in NUMERIC_FIELDS.items():
        if name in feature_names:
            fields.append({"name": name, "kind": "number", **details})

    for name, label in CATEGORICAL_FIELDS.items():
        prefix = f"{name}_"
        options = sorted(column[len(prefix):] for column in feature_names if column.startswith(prefix))
        if options:
            fields.append({"name": name, "label": label, "kind": "select", "options": options})

    return fields


def build_model_input(form_data: dict[str, str], feature_names: list[str], fields: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert browser values into the same one-hot encoded training format."""
    row: dict[str, float] = {feature: 0.0 for feature in feature_names}

    for field in fields:
        name = field["name"]
        value = form_data.get(name, "").strip()

        if not value:
            raise ValueError(f"Enter a value for {field['label']}.")

        if field["kind"] == "number":
            try:
                numeric_value = float(value)
            except ValueError as error:
                raise ValueError(f"{field['label']} must be a number.") from error

            if not field["min"] <= numeric_value <= field["max"]:
                raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
            row[name] = numeric_value
            continue

        if value not in field["options"]:
            raise ValueError(f"Choose a valid option for {field['label']}.")

        encoded_column = f"{name}_{value}"
        if encoded_column in row:
            row[encoded_column] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def app_context() -> dict[str, Any]:
    """Return values shared by all pages."""
    model_files = get_model_files()
    return {
        "models_ready": bool(model_files),
        "model_count": len(model_files),
        "model_options": [
            {"file": file.name, "key": file.stem, "label": humanize(file.stem)}
            for file in model_files
        ],
    }


@app.route("/")
def home() -> str:
    return render_template("index.html", **app_context())


@app.route("/predict", methods=["GET", "POST"])
def predict() -> str:
    context = app_context()

    try:
        feature_names = get_feature_names()
        fields = get_form_fields(feature_names)
    except (FileNotFoundError, OSError) as error:
        return render_template("predict.html", **context, error=str(error), fields=[], submitted={})

    context.update({"fields": fields, "submitted": request.form.to_dict()})
    if request.method == "GET":
        return render_template("predict.html", **context)

    selected_filename = request.form.get("model_file", "")
    allowed_models = {model["file"] for model in context["model_options"]}

    if selected_filename not in allowed_models:
        context["error"] = "Choose one of the saved models before predicting."
        return render_template("predict.html", **context)

    try:
        input_data = build_model_input(request.form.to_dict(), feature_names, fields)
        prediction = float(load_model(selected_filename).predict(input_data)[0])

        if not np.isfinite(prediction) or prediction < 0:
            raise ValueError("The selected model returned an invalid price.")

        area = float(request.form["Size_in_SqFt"])
        price_per_sqft = (prediction * 100000) / area if area else None
        return render_template(
            "result.html",
            **context,
            prediction=prediction,
            price_per_sqft=price_per_sqft,
            model_name=humanize(Path(selected_filename).stem),
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        context["error"] = str(error)
        return render_template("predict.html", **context)
    except Exception:
        context["error"] = "The prediction could not be completed. Check the saved model files and try again."
        return render_template("predict.html", **context)


@app.route("/models")
def models() -> str:
    context = app_context()
    context["catalogue"] = [
        {
            "name": humanize(model["key"]),
            "filename": model["file"],
            "description": MODEL_DESCRIPTIONS.get(model["key"], ("", "Saved regression model."))[1],
        }
        for model in context["model_options"]
    ]
    return render_template("models.html", **context)


@app.route("/about")
def about() -> str:
    return render_template("about.html", **app_context())


if __name__ == "__main__":
    app.run(debug=True)
