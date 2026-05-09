import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df


def encode(df):
    df['gender'] = (df['gender'] == 'Male').astype(int)

    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df


def preprocess(path):
    df = load_data(path)
    df = clean(df)
    df = encode(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler
