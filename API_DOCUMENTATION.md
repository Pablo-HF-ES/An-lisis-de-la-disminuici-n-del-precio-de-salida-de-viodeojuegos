# Video Games ML API Documentation

## Overview

The Video Games ML API provides REST endpoints for making predictions on video game launch price degradation using trained machine learning models. The API supports multiple model types and batch processing capabilities.

## Features

- **Multi-Model Support**: Choose between Linear, Ridge, Lasso, and ElasticNet regression models
- **Batch Processing**: Make predictions on multiple samples in a single request
- **Health Checks**: Built-in health check endpoints for monitoring
- **CORS Enabled**: Full Cross-Origin Resource Sharing support
- **Error Handling**: Comprehensive error messages and validation

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Pablo-HF-ES/An-lisis-de-la-disminuici-n-del-precio-de-salida-de-viodeojuegos.git
cd An-lisis-de-la-disminuici-n-del-precio-de-salida-de-viodeojuegos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (if not already trained):
```bash
python src/features.py
python src/train.py
```

### Running the API

#### Local Development

```bash
python app.py
```

The API will be available at `http://localhost:5000`

#### Docker Deployment

Build and run using Docker:
```bash
docker build -t video-games-ml-api .
docker run -p 5000:5000 video-games-ml-api
```

Or using docker-compose:
```bash
docker-compose up -d
```

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

Check if the API is running and models are loaded.

**Response**:
```json
{
  "status": "healthy",
  "available_models": ["linear", "ridge", "lasso", "elasticnet"]
}
```

**Status Codes**:
- `200`: API is healthy
- `503`: API is unhealthy (models not loaded)

---

### 2. Get Available Models

**Endpoint**: `GET /models`

Get list of available trained models.

**Response**:
```json
{
  "available_models": ["linear", "ridge", "lasso", "elasticnet"],
  "default_model": "ridge"
}
```

---

### 3. Single/Batch Prediction

**Endpoint**: `POST /predict`

Make predictions for one or multiple samples.

**Request Body** (Single Prediction):
```json
{
  "data": {
    "feature1": 100.5,
    "feature2": 50.2,
    "feature3": 75.8
  },
  "model": "ridge"
}
```

**Request Body** (Batch Prediction):
```json
{
  "data": [
    {
      "feature1": 100.5,
      "feature2": 50.2,
      "feature3": 75.8
    },
    {
      "feature1": 120.0,
      "feature2": 60.0,
      "feature3": 80.0
    }
  ],
  "model": "ridge"
}
```

**Response** (Single):
```json
{
  "prediction": 0.25,
  "model": "ridge",
  "status": "success"
}
```

**Response** (Batch):
```json
{
  "predictions": [
    {"index": 0, "prediction": 0.25},
    {"index": 1, "prediction": 0.28}
  ],
  "count": 2,
  "model": "ridge",
  "status": "success"
}
```

**Status Codes**:
- `200`: Predictions successful
- `400`: Invalid input data
- `503`: Model not available
- `500`: Server error

---

### 4. Batch Prediction (Dedicated Endpoint)

**Endpoint**: `POST /predict-batch`

Dedicated endpoint for batch predictions with detailed error handling.

**Request Body**:
```json
{
  "data": [
    {"feature1": 100.5, "feature2": 50.2},
    {"feature1": 120.0, "feature2": 60.0}
  ],
  "model": "ridge"
}
```

**Response**:
```json
{
  "predictions": [
    {"index": 0, "prediction": 0.25},
    {"index": 1, "prediction": 0.28}
  ],
  "total_samples": 2,
  "successful_predictions": 2,
  "errors": null,
  "model": "ridge",
  "status": "success"
}
```

---

### 5. API Information

**Endpoint**: `GET /info`

Get general API information and available endpoints.

**Response**:
```json
{
  "service": "Video Game Price Degradation Predictor",
  "version": "1.0.0",
  "description": "ML API for predicting video game launch price degradation",
  "endpoints": {
    "/health": "Health check endpoint",
    "/models": "List available models",
    "/predict": "Single or batch prediction (POST)",
    "/predict-batch": "Batch prediction (POST)",
    "/info": "API information"
  },
  "available_models": ["linear", "ridge", "lasso", "elasticnet"],
  "default_model": "ridge"
}
```

---

## Usage Examples

### Python Example

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Single prediction
payload = {
    "data": {
        "feature1": 100.5,
        "feature2": 50.2,
        "feature3": 75.8
    },
    "model": "ridge"
}

response = requests.post(url, json=payload)
print(response.json())
```

### cURL Example

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature1": 100.5,
      "feature2": 50.2,
      "feature3": 75.8
    },
    "model": "ridge"
  }'

# Get available models
curl http://localhost:5000/models

# Health check
curl http://localhost:5000/health
```

### JavaScript Example

```javascript
async function makePrediction() {
  const url = 'http://localhost:5000/predict';
  
  const payload = {
    data: {
      feature1: 100.5,
      feature2: 50.2,
      feature3: 75.8
    },
    model: 'ridge'
  };
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });
  
  const result = await response.json();
  console.log(result);
}
```

---

## Error Handling

The API returns detailed error messages with appropriate HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| 200 | Successful prediction |
| 400 | Invalid input data or missing required fields |
| 404 | Endpoint not found |
| 500 | Server error during prediction |
| 503 | Model or pipeline not available |

### Error Response Example

```json
{
  "error": "Invalid model type. Must be one of: ['linear', 'ridge', 'lasso', 'elasticnet']"
}
```

---

## Model Types

### Available Models

1. **Linear**: Simple linear regression
2. **Ridge**: Ridge regression with alpha tuning (alphas: [0.1, 1.0, 10.0])
3. **Lasso**: Lasso regression with alpha tuning (alphas: [0.1, 1.0, 10.0])
4. **ElasticNet**: Elastic Net regression with alpha and L1 ratio tuning

### Default Model

The default model is **Ridge** regression, which provides a good balance between bias and variance.

---

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_data.py -v

# Run only unit tests
pytest tests/ -m unit
```

---

## Performance Considerations

- **Batch Processing**: Use `/predict-batch` for multiple predictions to improve throughput
- **Model Selection**: Ridge regression is recommended for most use cases
- **Input Validation**: All inputs are validated before processing
- **Caching**: Models are cached in memory after first load

---

## Troubleshooting

### Models Not Found

Ensure models have been trained and are located in the `models/` directory:
- `modelo_linear.joblib`
- `modelo_ridge.joblib`
- `modelo_lasso.joblib`
- `modelo_elasticnet.joblib`
- `preprocessing_pipeline.joblib`

### 503 Service Unavailable

This indicates models or pipeline are not loaded. Check:
1. Models directory exists and contains all required files
2. File permissions are correct
3. Check API logs for detailed error messages

### Feature Mismatch Errors

Ensure input features match the model training features. The preprocessing pipeline expects the same features that were used during training.

---

## Development

### Project Structure

```
.
├── app.py              # Main Flask API
├── setup.py            # Package setup
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── pytest.ini          # Pytest configuration
├── src/
│   ├── data_ingestion.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── models/             # Trained models and preprocessing pipeline
├── data/               # Data in different stages
└── tests/
    ├── test_data.py
    └── test_model.py
```

### Contributing

1. Create a feature branch
2. Write tests for new functionality
3. Run test suite to ensure all tests pass
4. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Support

For issues, questions, or contributions, please visit the GitHub repository:
https://github.com/Pablo-HF-ES/An-lisis-de-la-disminuici-n-del-precio-de-salida-de-viodeojuegos
