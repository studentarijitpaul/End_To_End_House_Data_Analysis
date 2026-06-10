House-Price-Prediction
==============================
Overview
This project focuses on predicting house prices in India using machine learning techniques. The model was trained on a subset of the India Housing Prices Dataset from Kaggle. This dataset provides detailed insights into housing market trends across various Indian states, including property types, pricing, location, and amenities.

Key Features:

Data Exploration and Preprocessing: Includes exploratory data analysis (EDA) to understand the data and preprocessing steps to prepare it for modeling.
Model Training: Employs various regression models including Linear Regression, Decision Tree, Random Forest, and XGBoost.
Hyperparameter Tuning: Parameter tuning was performed for all models, with a focus on Grid Search for optimizing XGBoost.
Performance Evaluation: Comprehensive evaluation of each model using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).
Dataset
The model was trained using the India Housing Prices Dataset available on Kaggle.

The India Housing Prices Dataset contains 2.5 lakh rows and 23 columns, offering detailed insights into housing market trends across various Indian states. It includes attributes related to property types, pricing, location, and amenities, making it suitable for analysis ranging from basic descriptive statistics to advanced machine learning applications.

Note: Due to computational limitations, this project utilized a sample of approximately 100,000 rows from the original dataset.

Business Applications
This house price prediction model can be valuable for:

🏘️ Real Estate Agents: Quickly estimate property values based on location and features
💰 Property Investors: Make data-driven decisions for investment opportunities across Indian states
🏦 Banks & Lenders: Assist in property valuation for mortgage and loan assessments
🏗️ Property Developers: Analyze market trends and optimize pricing strategies for new developments
🏡 Home Buyers: Get fair price estimates before making purchase decisions
📊 Market Analysts: Study housing market trends and price patterns across different Indian regions
The model's high accuracy (99.6% R² score with XGBoost) makes it a reliable tool for stakeholders in the Indian real estate market to make informed decisions.

Methodology
The following steps were undertaken in this project:

Exploratory Data Analysis (EDA): Initial exploration of the dataset to understand its structure, identify patterns, and gain insights into the features.
Data Preprocessing:
Dropping Unnecessary Columns: Identification and removal of irrelevant or redundant columns.
Ordinal Encoding: Encoding of ordinal categorical features into numerical representations.
Data Sampling: Selection of a subset of approximately 100,000 rows from the dataset due to limited computational resources.
One-Hot Encoding: Application of One-Hot Encoding using DictVectorizer to convert remaining categorical features into a numerical format suitable for machine learning models.
Model Training and Evaluation:
Training of the following regression models:
Linear Regression
Decision Tree
Random Forest
XGBoost
Parameter tuning for each model to optimize performance.
Grid Search was specifically used to fine-tune the hyperparameters of the XGBoost model.
Performance Evaluation: Evaluation of each trained model using the following metrics:
Mean Absolute Error (MAE): Average absolute difference between the predicted and actual prices.
Mean Squared Error (MSE): Average of the squared differences between the predicted and actual prices.
Root Mean Squared Error (RMSE): Square root of the MSE, providing an error metric in the original unit of the target variable.
R-squared (R2): A statistical measure representing the proportion of the variance in the dependent variable that is predictable from the independent variables.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
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
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
