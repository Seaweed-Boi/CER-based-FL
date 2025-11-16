"""
Generate realistic bank fraud detection dataset for FL training.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)

# Generate base features
n_samples = 5000
n_features = 20

# Create imbalanced classification dataset
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=15,
    n_redundant=3,
    n_clusters_per_class=2,
    weights=[0.95, 0.05],  # 5% fraud rate
    flip_y=0.01,
    random_state=42
)

# Create realistic feature names
feature_names = [
    'transaction_amount',
    'transaction_hour',
    'transaction_day',
    'account_age_days',
    'num_transactions_24h',
    'avg_transaction_amount',
    'transaction_velocity',
    'merchant_category',
    'is_international',
    'card_present',
    'billing_zip_match',
    'shipping_zip_match',
    'email_domain_age',
    'ip_address_risk_score',
    'device_fingerprint_match',
    'transaction_type_code',
    'currency_code',
    'payment_method',
    'customer_risk_score',
    'merchant_risk_score'
]

# Scale features to realistic ranges
X_scaled = X.copy()
X_scaled[:, 0] = np.abs(X[:, 0]) * 1000 + 50  # transaction_amount: 50-5000
X_scaled[:, 1] = (X[:, 1] + 3) / 6 * 24  # transaction_hour: 0-24
X_scaled[:, 2] = (X[:, 2] + 3) / 6 * 7  # transaction_day: 0-7
X_scaled[:, 3] = np.abs(X[:, 3]) * 365 + 1  # account_age_days: 1-1000+
X_scaled[:, 4] = np.abs(X[:, 4]) * 10  # num_transactions_24h: 0-30
X_scaled[:, 14] = (X[:, 14] + 3) / 6  # device_fingerprint_match: 0-1

# Create DataFrame
df = pd.DataFrame(X_scaled, columns=feature_names)
df['is_fraud'] = y

# Save to CSV
df.to_csv('bank_data.csv', index=False)

print(f"✅ Generated bank fraud dataset:")
print(f"   Total samples: {len(df)}")
print(f"   Features: {len(feature_names)}")
print(f"   Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"   Normal cases: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.2f}%)")
print(f"\n📊 Sample data:")
print(df.head())
print(f"\n💾 Saved to: bank_data.csv")
