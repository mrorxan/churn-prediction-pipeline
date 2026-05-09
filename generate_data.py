import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
n = 7043

tenure = rng.randint(0, 73, n)

contract = rng.choice(
    ['Month-to-month', 'One year', 'Two year'],
    n, p=[0.55, 0.21, 0.24]
)

monthly = np.round(rng.uniform(18.5, 118.75, n), 2)

total = np.round(tenure * monthly * rng.uniform(0.96, 1.04, n), 2)
total_str = [str(t) if t > 0 else ' ' for t, t in zip(total, tenure)]

churn_prob = np.where(
    contract == 'Month-to-month',
    np.clip(0.43 - 0.003 * tenure, 0.15, 0.55),
    np.where(
        contract == 'One year',
        np.clip(0.12 - 0.001 * tenure, 0.02, 0.15),
        np.clip(0.04 - 0.0005 * tenure, 0.01, 0.06)
    )
)
churn = rng.binomial(1, churn_prob)

gender = rng.choice(['Male', 'Female'], n)
senior = rng.binomial(1, 0.16, n)
partner = rng.choice(['Yes', 'No'], n, p=[0.48, 0.52])
dependents = rng.choice(['Yes', 'No'], n, p=[0.30, 0.70])
phone_service = rng.choice(['Yes', 'No'], n, p=[0.90, 0.10])

multiple_lines = np.where(
    phone_service == 'No',
    'No phone service',
    rng.choice(['Yes', 'No'], n, p=[0.42, 0.58])
)

internet = rng.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
no_internet = internet == 'No'

def inet_feature():
    vals = rng.choice(['Yes', 'No'], n, p=[0.44, 0.56])
    return np.where(no_internet, 'No internet service', vals)

paperless = rng.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment = rng.choice(
    ['Electronic check', 'Mailed check',
     'Bank transfer (automatic)', 'Credit card (automatic)'],
    n, p=[0.34, 0.23, 0.22, 0.21]
)

chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
cids = [
    f"{rng.randint(1000, 9999)}-{''.join(rng.choice(chars, 5))}"
    for _ in range(n)
]

df = pd.DataFrame({
    'customerID': cids,
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet,
    'OnlineSecurity': inet_feature(),
    'OnlineBackup': inet_feature(),
    'DeviceProtection': inet_feature(),
    'TechSupport': inet_feature(),
    'StreamingTV': inet_feature(),
    'StreamingMovies': inet_feature(),
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total_str,
    'Churn': np.where(churn == 1, 'Yes', 'No'),
})

df.to_csv('data/telco_churn.csv', index=False)
print(f'Generated {n} rows. Churn rate: {churn.mean():.3f}')
print(df['Churn'].value_counts())
