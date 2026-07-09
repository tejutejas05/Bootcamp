import mlflow 
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# load data 
data = load_breast_cancer()
X , y = data.data , data.target

#split data
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2,random_state=42)

mlflow.set_experiment("Breast_Cancer_Prediction")

depths = [2,4,6,8,10]

for depth in depths:
    with mlflow.start_run(run_name=f"Random Forest (depth={depth})"):
        model = RandomForestClassifier(max_depth=depth,random_state=42)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)

        mlflow.log_param("max_depth",depth)
        mlflow.log_metric("accuracy",accuracy)
        
        print(f"Depth: {depth}, Accuracy: {accuracy}")

        mlflow.log_param("model_name","Random Forest")
        mlflow.log_metric("accuracy",accuracy)

        disp = ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        #Log model
        mlflow.sklearn.log_model(model,"Random Forest")

        print(f"""
        Run completed for max_depth={depth}
        """)

import joblib

model = joblib.load("model.pkl")
