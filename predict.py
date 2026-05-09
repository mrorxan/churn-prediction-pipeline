import joblib
from sklearn.metrics import classification_report

from src.preprocess import preprocess


# Run main.py first to train and save models before using this script.

model = joblib.load('models/gradient_boosting.pkl')

_, X_test, _, y_test, _ = preprocess('data/telco_churn.csv')

sample = X_test.iloc[:10]
preds = model.predict(sample)
actuals = y_test.iloc[:10].values

print('Sample predictions (Gradient Boosting):')
print(f'{"#":<4} {"Predicted":<14} {"Actual":<10}')
print('-' * 30)
for i, (pred, actual) in enumerate(zip(preds, actuals)):
    pred_label = 'Churn' if pred == 1 else 'No Churn'
    actual_label = 'Churn' if actual == 1 else 'No Churn'
    print(f'{i+1:<4} {pred_label:<14} {actual_label:<10}')
