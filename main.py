import os
import joblib
from sklearn.metrics import classification_report

from src.preprocess import preprocess
from src.train import get_models, train_and_save
from src.evaluate import evaluate, plot_confusion_matrix


DATA_PATH = 'data/telco_churn.csv'

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('Preprocessing...')
X_train, X_test, y_train, y_test, scaler = preprocess(DATA_PATH)
joblib.dump(scaler, 'models/scaler.pkl')
print(f'  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples')

print('\nTraining models...')
models = get_models()
trained = train_and_save(models, X_train, y_train)

print('\nResults:')
print('-' * 50)
for name, model in trained.items():
    preds, acc, f1 = evaluate(model, X_test, y_test)
    plot_confusion_matrix(y_test, preds, name)
    print(f'\n{name}')
    print(f'  Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}')
    print(classification_report(y_test, preds, target_names=['No Churn', 'Churn']))
