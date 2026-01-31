from transformers import BertForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
import re
import string
import torch
from typing import List, Dict
from googletrans import Translator
from langdetect import detect

translator = Translator()


# Configuration
MODEL_PATH = "/quntizedbert-clf/"
TOKENIZER_PATH = "/quntizedbert-clf/"

LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]

class SentimentRequest(BaseModel):
    text: str

class SentimentResult(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    predictions: List[SentimentResult]
    top_prediction: SentimentResult


# Load model and tokenizer
def load_model_and_tokenizer():
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model.eval()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train.py first to train the model")
        raise


# Initialize model at module load
try:
    model, tokenizer, device = load_model_and_tokenizer()
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    model, tokenizer, device = None, None, None


def preprocess(text: str) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"


async def predict_sentiment(text: str) -> Dict[str, any]:
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Please run train.py first.")

    language = detect_language(text)

    if language != "en":
        # Translate to English
        translated = translator.translate(text, dest="en")
        text = translated.text  # use translated text


    cleaned_text = preprocess(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)

    predictions = []
    for label, prob in zip(LABELS, probs.cpu().tolist()):
        predictions.append({
            "label": label,
            "score": round(prob, 4)
        })

    predictions.sort(key=lambda x: x["score"], reverse=True)

    return {
        "predictions": predictions,
        "top_prediction": predictions[0]
    }


def get_top_prediction(text: str) -> Dict[str, any]:
    """Get only the top prediction"""
    results = predict_sentiment(text)
    return results[0]