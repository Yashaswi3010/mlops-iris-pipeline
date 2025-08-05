# scripts/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, X_train, y_train, model_name, params={}):
    """Train a model and log it with MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Instantiate and train the model
        clf = model(**params)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_train)

        # Log parameters and metrics
        mlflow.log_params(params)
        accuracy = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(clf, model_name)

        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        return clf

def main():
    """Main function to run the training pipeline."""
    print("--- Starting Model Training ---")

    # Set tracking URI (can be a local folder, HTTP server, or Databricks)
    # MLflow will create the `mlruns` directory locally if not specified.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Iris Classification")

    # Load data
    iris_df = pd.read_csv("data/raw/iris.csv")
    X = iris_df.drop("target", axis=1)
    y = iris_df["target"]

    # For simplicity, we train on the full dataset. In a real scenario, use a train/test split.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    train_model(LogisticRegression, X, y, "LogisticRegression", params={"max_iter": 200})

    # Train Random Forest
    train_model(RandomForestClassifier, X, y, "RandomForest", params={"n_estimators": 100, "max_depth": 5})

    print("--- Model Training Complete ---")

if __name__ == '__main__':
    main()