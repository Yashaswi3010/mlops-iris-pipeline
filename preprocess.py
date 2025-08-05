# scripts/preprocess.py
import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

def main():
    """Fetches the Iris dataset and saves it to the data/raw directory."""
    print("--- Preprocessing Data ---")

    # Define paths
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Load Iris dataset from sklearn
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target

    # Save to csv
    iris_df.to_csv(raw_data_path / "iris.csv", index=False)

    print(f"Dataset saved to {raw_data_path / 'iris.csv'}")
    print("--- Preprocessing Complete ---")

if __name__ == '__main__':
    main()