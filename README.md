# Multilingual Aspect-Based Sentiment Analysis

A complete Multilingual Aspect-Based sentiment analysis application using BERT (Bidirectional Encoder Representations from Transformers) for multi-class text classification.

## 🎯 Features

- **Multi-class Classification**: Classifies text into 4 categories (Positive, Negative, Neutral, Irrelevant)
- **BERT-based Model**: Uses state-of-the-art transformer architecture
- **Web Interface**: Beautiful, responsive UI for easy interaction
- **REST API**: Complete API for programmatic access
- **End-to-End Pipeline**: From data loading to deployment

## 📋 Project Structure

```
Multilang-sentiment-analysis/
├── src/
│   ├── __init__.py          # Makes 'src' a package
│   ├── model.py             # Inference and model loading
│   └── app.py               # FastAPI application
├── static/
│   └── style.css            # CSS, JS, and images
├── templates/
│   └── index.html           # Jinja2 templates for UI
├── notebooks/               # Keep experimental files separate
│   └── testing.ipynb
├── Dockerfile               # Root level for easy building
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```


## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/pritc2611/Multilang-aspect-based-sentiment-analysis
```

```bash
cd Multilang-aspect-based-sentiment-analysis
```

```bash
pip install -r requirments.txt
```

```bash
cd src
```

### 2. Run the Application

```bash
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
    {"label": "Conflict", "score": 0.0050}
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
- **Classes**: 4 (Positive, Negative, Neutral, Conflict)
- **Max Sequence Length**: No Limits

### Training
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
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

| Endpoint      | Method | Description         |
|---------------|--------|---------------------|
| `/`           |  GET   | Web interface       |
| `/predict`    |  POST  | Web form prediction |
| `/api/predict`|  POST  | REST API prediction |
| `/api/health` |  GET   | Health check        |
| `/api/labels` |  GET   | Available labels    |


**Happy Analyzing! 🎭**
