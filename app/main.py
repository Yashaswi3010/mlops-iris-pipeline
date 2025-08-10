# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import logging
from collections import Counter

from pydantic.fields import Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- In-memory monitoring ---
# In a real system, use Prometheus/Grafana
REQUEST_COUNTER = Counter()
PREDICTION_COUNTER = Counter()
# ----------------------------

# Define the input data schema using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm, must be positive.")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm, must be positive.")
    petal_length: float = Field(..., gt=0, description="Petal length in cm, must be positive.")
    petal_width: float = Field(..., gt=0, description="Petal width in cm, must be positive.")

# Initialize FastAPI app
app = FastAPI(title="Iris Classifier API", version="1.0.0")

Instrumentator().instrument(app).expose(app)

# Load the production model from MLflow Model Registry
# Ensure the MLflow UI is running or you have a tracking server configured
MODEL_NAME = "iris-classifier"
MODEL_STAGE = "Production"
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    logger.info(f"Successfully loaded model '{MODEL_NAME}' version '{MODEL_STAGE}'")
except Exception as e:
    model = None
    logger.error(f"Error loading model: {e}")

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to check API status."""
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict", tags=["Prediction"])
def predict(features: IrisFeatures):
    """Endpoint to make a prediction."""
    REQUEST_COUNTER.update({"total": 1})
    if model is None:
        logger.error("Model not loaded. Prediction cannot be made.")
        return {"error": "Model not loaded. Please check server logs."}

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([features.dict()])
        # Rename columns to match model's expected feature names
        input_df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

        logger.info(f"Received prediction request: {features.dict()}")

        # Make prediction
        prediction = model.predict(input_df)
        prediction_label = int(prediction[0])

        PREDICTION_COUNTER.update({str(prediction_label): 1})
        logger.info(f"Prediction result: {prediction_label}")

        return {"prediction": prediction_label}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNTER.update({"failed": 1})
        return {"error": str(e)}