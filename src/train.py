import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models():
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
    }


def train_and_save(models, X_train, y_train, save_dir='models'):
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'{save_dir}/{name}.pkl')
        trained[name] = model
        print(f'  {name}: done')
    return trained
