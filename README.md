House-Price-Prediction
==============================
Overview
This project focuses on predicting house prices in India using machine learning techniques. The model was trained on a subset of the India Housing Prices Dataset from Kaggle. This dataset provides detailed insights into housing market trends across various Indian states, including property types, pricing, location, and amenities.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── dat
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Web Application
------------

The trained models are available through a multi-page Flask web application:

- `/` - project home page
- `/predict` - select a saved model and estimate a property price
- `/models` - browse the saved regression models
- `/about` - project and prediction workflow

Run the website locally from the project root:

```powershell
python -m pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:5000` in your browser. The application reads `Models/features.pkl` and the saved `.pkl` model files directly from the existing `Models` folder.

For Render deployment, the repository includes `render.yaml`. The production start command is `gunicorn server:app`.
