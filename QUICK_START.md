# Quick Start Guide - Video Games ML API

## 🚀 Quick Start (5 minutes)

### Option 1: Docker (Easiest)
```bash
# Start the API with Docker Compose
docker-compose up -d

# Check if it's running
curl http://localhost:5000/health

# Test a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": 100.5, "feature2": 50.2}, "model": "ridge"}'
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure models are trained (check models/ directory first)
# If not, run:
# python src/features.py
# python src/train.py

# Start the API
python app.py

# In another terminal, test it
curl http://localhost:5000/health
```

## 📝 Common Use Cases

### 1. Get API Info
```bash
curl http://localhost:5000/info
```

### 2. Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"feature1": 100, "feature2": 50, "feature3": 75},
    "model": "ridge"
  }'
```

### 3. Batch Predictions
```bash
curl -X POST http://localhost:5000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"feature1": 100, "feature2": 50},
      {"feature1": 120, "feature2": 60}
    ],
    "model": "ridge"
  }'
```

### 4. List Available Models
```bash
curl http://localhost:5000/models
```

## 🧪 Run Tests
```bash
# All tests with coverage
pytest

# Specific test file
pytest tests/test_data.py -v

# With coverage report
pytest --cov=src --cov-report=html
```

## 📚 Full Documentation
See `API_DOCUMENTATION.md` for complete API reference and examples.

## 🔧 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'src'"**
A: Make sure you're running from project root and have pip installed the dependencies

**Q: "503 Service Unavailable"**
A: Ensure models are trained and present in `models/` directory:
- modelo_linear.joblib
- modelo_ridge.joblib
- modelo_lasso.joblib
- modelo_elasticnet.joblib
- preprocessing_pipeline.joblib

**Q: Docker container won't start**
A: Check logs with `docker-compose logs api` and ensure port 5000 is not in use

## 📞 Support
For issues, see API_DOCUMENTATION.md or check the repository on GitHub.
