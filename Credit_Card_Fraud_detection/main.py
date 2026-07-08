import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv("creditcard_cleaned.csv")
print(df.head())
print(df.columns)

#train_test_split
X = df.drop("class", axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_

important_feature = X_train.columns[importances.argsort()[-10:]]
X_train_selected = X_train[important_feature]
X_test_selected = X_test[important_feature]
print(X_train_selected.head())

retrain = RandomForestClassifier(random_state=42)
retrain.fit(X_train_selected, y_train)

y_pred = retrain.predict(X_test_selected)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print(f"Classification Report: ",{classification_report(y_test,y_pred)})


joblib.dump(retrain, "creditcard_model.pkl")
print("Model saved successfully")