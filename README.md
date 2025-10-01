# Credit Card Fraud Detection using Machine Learning

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Data Preprocessing](#data-preprocessing)
* [Feature Engineering](#feature-engineering)
* [Modeling](#modeling)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [How to Run](#how-to-run)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Deployment (Optional)](#deployment-optional)
* [Reproducibility](#reproducibility)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Project Overview

This project detects fraudulent credit card transactions using machine learning. The aim is to build a pipeline that performs data cleaning, feature engineering, model training, evaluation, and (optionally) deployment. Due to the extreme class imbalance typical of fraud datasets, the pipeline focuses on robust evaluation and techniques that mitigate imbalance (sampling, class weights, anomaly detection approaches).

Key goals:

* Explore and understand the dataset
* Build and compare multiple models (logistic regression, tree-based models, and ensemble methods)
* Handle class imbalance and measure model performance using appropriate metrics
* Provide an easy-to-run notebook and scripts for reproducibility

---

## Dataset

* **Source:** Commonly used dataset is the "Credit Card Fraud Detection" dataset derived from European card transactions (commonly found on Kaggle). If you use another dataset, update `data/README.md`.
* **Format:** CSV (transactions as rows, features as columns)
* **Target column:** `Class` (0 = legitimate, 1 = fraud)

**Important note:** The original Kaggle dataset contains PCA-transformed features named `V1..V28`, `Time`, and `Amount`. If you use a different dataset, adapt preprocessing steps accordingly.

---

## Exploratory Data Analysis (EDA)

* Check class distribution and quantify the imbalance.
* Visualize distributions of `Amount` and (if present) `Time`.
* Correlation heatmap of features (careful with PCA-transformed features — they may not be directly interpretable).
* Identify missing values and outliers.

Typical EDA outcomes:

* Very small fraction of transactions are fraud (e.g., <0.5%).
* `Amount` often skewed — consider log or standard scaling.

---

## Data Preprocessing

1. **Missing values:** Impute or drop if present.
2. **Scaling:** Use `StandardScaler` for features like `Amount` and `Time` (if not PCA). For neural networks, scaling all features to zero mean and unit variance helps.
3. **Train/test split:** Use a stratified split to maintain class proportions (e.g., `train_test_split(..., stratify=y, test_size=0.2)`).
4. **Sampling or weighting:** Options to handle imbalance:

   * **Class weights** in model (e.g., `class_weight='balanced'` for scikit-learn models)
   * **Undersampling** majority class (with care — may lose information)
   * **Oversampling** minority class (SMOTE, ADASYN)
   * **Ensemble of balanced subsets** (e.g., EasyEnsemble)

---

## Feature Engineering

* If using raw features, create interaction terms or aggregates if meaningful.
* Time-derived features: hour of day, transaction age, etc.
* Amount transformation: `log(Amount + 1)` or robust scaling.
* Remove or transform features that leak the label (if any).

---

## Modeling

Try multiple model families and compare results:

### Baseline models

* Logistic Regression (with `class_weight='balanced'`)
* Decision Tree (depth-limited)

### Stronger models

* Random Forest
* XGBoost / LightGBM (often strong on tabular data)
* Gradient Boosting (scikit-learn `HistGradientBoostingClassifier`)

### Anomaly detection / Unsupervised

* Isolation Forest
* One-Class SVM

### Neural Networks

* Simple MLP with dropout, batch normalization, and balanced batches or weighted loss

Model selection tips:

* Use cross-validation with stratification (e.g., `StratifiedKFold`).
* When using oversampling like SMOTE, apply it inside cross-validation pipeline to avoid leakage.

---

## Evaluation Metrics

Because of class imbalance, accuracy is not useful. Prefer:

* **Precision, Recall, F1-score** (for the fraud class)
* **ROC AUC** and **PR AUC (Precision-Recall AUC)** — PR AUC is more informative for imbalanced data
* **Confusion matrix** at chosen probability threshold
* **Cost-sensitive analysis**: compute expected monetary loss saved by the model using business costs (false positives vs false negatives)

Report metrics for both validation and held-out test set.

---

## Results

Include a short summary of results and the best performing model(s), for example:

* Model: `XGBoost` with SMOTE in training pipeline
* Best metric: PR-AUC = `0.45` (example)
* Precision at recall 0.75: `0.12` (example)

Add visualizations:

* ROC curve
* Precision-Recall curve
* Feature importance (SHAP values if possible)

---

## How to Run

### Option A — Jupyter Notebook (recommended)

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Start notebook:

```bash
jupyter lab  # or jupyter notebook
```

3. Open `notebooks/credit_card_fraud_detection.ipynb` and run cells in order.

### Option B — Run scripts

* Preprocessing and training scripts in `src/` can be executed as:

```bash
python src/train.py --config config/train_config.yaml
python src/evaluate.py --model outputs/best_model.pkl
```

### Configuration

* Use `config/train_config.yaml` to set model hyperparameters, sampling strategy, and data paths.

---

## Repository Structure

```
├── data/
│   ├── raw/                  # raw CSVs (not tracked in git)
│   └── processed/            # processed datasets
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── outputs/
│   ├── models/
│   └── figures/
├── requirements.txt
├── README.md
└── config/
    └── train_config.yaml
```

---

## Requirements

A minimal `requirements.txt` example:

```
python>=3.8
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
imbalanced-learn
jupyterlab
shap

# optional for deployment
fastapi
uvicorn
joblib
```

---

## Deployment (Optional)

* Export the trained model using `joblib` or `pickle`.
* Wrap with a small REST API (FastAPI/Flask) that accepts a transaction payload and returns a fraud probability and explanation.
* Add rate-limiting and input validation; run model inference asynchronously if throughput requires.
* Consider monitoring model drift and setting up periodic re-training.

---

## Reproducibility

* Set random seeds (`numpy`, `random`, and `scikit-learn`) in training scripts.
* Log experiments with MLflow or Weights & Biases for experiment tracking.
* Save version of dataset and exact `requirements.txt` (or use `pip freeze`).

---

## Contributing

Contributions are welcome. Please:

1. Fork the repo
2. Create a feature branch
3. Submit a PR with tests and clear description

---

## License

Specify a license (e.g., MIT). Update `LICENSE` file accordingly.

---

## Contact

Maintainer: Your Name (`your.email@example.com`)

---

*Notes & Tips*

* Always be careful about data privacy and ensure the dataset is used according to its license.
* If working with real transactional data, consult legal and compliance teams before using or sharing data.
