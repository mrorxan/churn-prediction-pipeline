# Customer Churn Prediction

A machine learning pipeline for predicting customer churn on telecom data. The goal was to build a clean, reproducible workflow covering preprocessing through model evaluation — not to maximize accuracy at all costs.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features). The data includes contract type, tenure, monthly charges, and a range of service subscriptions. Target variable is binary: whether the customer churned or not.

Download the CSV from Kaggle and place it at `data/telco_churn.csv`. Alternatively, `generate_data.py` can produce a synthetic version of the same schema for quick testing.

The dataset has a class imbalance (~27% churn), which affects how the models are configured and evaluated.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline (preprocessing → training → evaluation):

```bash
python main.py
```

Run inference on a sample of test customers:

```bash
python predict.py
```

## Results

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Logistic Regression | 0.7381 | 0.7519 |
| Random Forest | 0.7942 | 0.7852 |
| Gradient Boosting | 0.7977 | 0.7890 |

Logistic Regression was configured with `class_weight='balanced'`, which improves churn recall significantly (0.78) compared to tree-based models, but at the cost of overall accuracy (0.74) — a reasonable trade-off depending on the business use case (e.g., catching more potential churners). Random Forest and Gradient Boosting achieve better overall accuracy (~0.80) and F1 scores, but they tend to predict 'No Churn' too conservatively, resulting in a lower churn recall (~0.51). There is a real tension here between precision and recall for the minority class that hyperparameter tuning would help address.

Confusion matrix plots for each model are saved to `results/`.

## Project Structure

```
├── data/              # Dataset (CSV)
├── notebooks/         # EDA notebook
├── src/
│   ├── preprocess.py  # Cleaning, encoding, scaling, train/test split
│   ├── train.py       # Model definitions and training
│   └── evaluate.py    # Metrics and confusion matrix
├── models/            # Saved model files (.pkl)
├── results/           # Confusion matrix plots
├── main.py            # Run full pipeline
└── predict.py         # Run inference on test samples
```

## What I'd improve next

- Cross-validated hyperparameter search (GridSearchCV) for Random Forest and Gradient Boosting
- Feature importance analysis to understand which features actually drive churn
- Try SMOTE or other resampling techniques to handle class imbalance more explicitly
- Threshold tuning on predicted probabilities instead of using the default 0.5 cutoff
