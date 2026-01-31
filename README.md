# Multi-language Sentiment Analysis

A complete Multi-language sentiment analysis application using BERT (Bidirectional Encoder Representations from Transformers) for multi-class text classification.

## 🎯 Features

- **Multi-class Classification**: Classifies text into 4 categories (Positive, Negative, Neutral, Irrelevant)
- **BERT-based Model**: Uses state-of-the-art transformer architecture
- **Web Interface**: Beautiful, responsive UI for easy interaction
- **REST API**: Complete API for programmatic access
- **End-to-End Pipeline**: From data loading to deployment

## 📋 Project Structure

```
Multilang-sentiment-analysis/
│
├── model.py                 # Inference and model loading
├── app.py                   # FastAPI application
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── templates/
│   └── index.html          # Web UI template
│
├── static/
│   └── style.css           # Styling
│
<<<<<<< HEAD
└── finetined-BERT/         # Trained model
=======
└── models/                 # Saved models (created after training)
    ├── quntizedbert-clf/  # Trained model
>>>>>>> 920b14fb3f180d4812cf8e08f2f80256a084e18d
```

## 🚀 Quick Start

### 1. Installation

```bash
mkdir Multilang-sentiment-analysis
cd Multilang-sentiment-analysis
```

```bash
git clone https://github.com/pritc2611/Multilang-sentiment-analysis
```

```bash
# Install dependencies
pip install -r requirments.txt
```

### 2. Run the Application

```bash
# use uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **Health Check**: http://localhost:8000/api/health

## 💻 Usage

### Web Interface

1. Navigate to http://localhost:8000
2. Enter text in the textarea
3. Click "Analyze Sentiment"
4. View predictions with confidence scores

### REST API

**Predict Sentiment:**
```bash
curl.exe -X POST http://localhost:8000/api/predict `
  -H "Content-Type: application/json" `
  -d '{"text":"Este producto es increíble"}'
```

**Response:**
```json
{
  "predictions": [
    {"label": "Positive", "score": 0.9245},
    {"label": "Neutral", "score": 0.0523},
    {"label": "Negative", "score": 0.0182},
    {"label": "Irrelevant", "score": 0.0050}
  ],
  "top_prediction": {
    "label": "Positive",
    "score": 0.9245
  }
}
```

**Get Available Labels:**
```bash
curl http://localhost:8000/api/labels
```

## 📊 Model Details

### Architecture
- **Base Model**: BERT (bert-base-uncased)
- **Task**: Sequence Classification
- **Classes**: 4 (Positive, Negative, Neutral, Irrelevant)
- **Max Sequence Length**: No Limits

### Training
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3 (with early stopping)
- **Metrics**: Accuracy, F1, Precision, Recall

### Preprocessing
- HTML tag removal
- URL removal
- Punctuation removal
- Lowercase conversion
- @ and # symbol removal

## 📦 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Web form prediction |
| `/api/predict` | POST | REST API prediction |
| `/api/health` | GET | Health check |
| `/api/labels` | GET | Available labels |



**Happy Analyzing! 🎭**
