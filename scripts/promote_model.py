# scripts/promote_model.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    """
    Compares the latest Staging model to the Production model.
    If the Staging model is better, it is promoted to Production.
    """
    print("--- Starting Model Promotion ---")
    
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    
    model_name = "iris-classifier"

    # Load a hold-out test set for unbiased evaluation
    data = pd.read_csv("data/raw/iris.csv")
    _, X_test, _, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)
    
    # Get latest Staging and Production models
    try:
        staging_version = client.get_latest_versions(model_name, stages=["Staging"])[0]
        prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    except IndexError:
        print("No model in Staging or Production. Exiting.")
        return

    print(f"Comparing Staging version {staging_version.version} with Production version {prod_version.version}")
    
    # Load models
    staging_model = mlflow.pyfunc.load_model(staging_version.source)
    prod_model = mlflow.pyfunc.load_model(prod_version.source)
    
    # Evaluate models
    staging_accuracy = accuracy_score(y_test, staging_model.predict(X_test))
    prod_accuracy = accuracy_score(y_test, prod_model.predict(X_test))
    
    print(f"Staging Model Accuracy: {staging_accuracy:.4f}")
    print(f"Production Model Accuracy: {prod_accuracy:.4f}")
    
    # Promote if staging model is better
    if staging_accuracy > prod_accuracy:
        print(f"Staging model is better. Promoting version {staging_version.version} to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        print("Production model is better or equal. Keeping current Production model.")
        # Optionally, archive the staging model
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version.version,
            stage="Archived"
        )
        
    print("--- Model Promotion Complete ---")

if __name__ == '__main__':
    main()