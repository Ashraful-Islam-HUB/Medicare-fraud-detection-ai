# Train fraud detection model

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# sample data load
data = pd.read_csv('data/sample_claims.csv')

X = data[['claim_amount']]
y = data['fraud_label']

model = RandomForestClassifier()
model.fit(X, y)

print("Model trained successfully")
