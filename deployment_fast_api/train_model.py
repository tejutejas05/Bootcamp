import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data.csv")

X = df[['Age', 'Salary']]
y = df['Approved']

model = RandomForestClassifier(random_state=42)

model.fit(X,y)

joblib.dump(model, "loan_model.pkl")

print("Training Completed, and Model is saved as loan_model.pkl")

