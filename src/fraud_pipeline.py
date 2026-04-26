import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data():
    return pd.read_csv('data/sample_claims.csv')

def preprocess(data):
    return data[['claim_amount']], data['fraud_label']

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def evaluate(model, X, y):
    predictions = model.predict(X)
    print(classification_report(y, predictions))

if __name__ == "__main__":
    data = load_data()
    X, y = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
