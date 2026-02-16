from transformers import BertForSequenceClassification, AutoTokenizer , pipeline
from pydantic import BaseModel
import re
import string
import torch
from typing import List, Dict , Optional
from googletrans import Translator
from langdetect import detect

translator = Translator()


# Configuration
TOKEN_CLF_MODEL = "./aspect-token-clf/"
TEXT_CLF_MODEL = "./aspect-sentimentV2-clf/"

LABELS = ['Negative','Neutral','Positive','Conflict']

class SentimentRequest(BaseModel):
    text: str

class SentimentResult(BaseModel):
    label: str
    score: float

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float

class SentimentResponse(BaseModel):
    predictions: List[SentimentResult]
    top_prediction: SentimentResult
    aspects: List[AspectResult]  # Use the specific Aspect model here

# Load model and tokenizer
def load_model_and_tokenizer():
    try:
        aspect_extractor = pipeline("token-classification",model="CPDT/aspect-extrector-Bert",aggregation_strategy="simple")
        aspect_sentiment = pipeline("text-classification", model="CPDT/aspect-sentiment-clf")

        sentiment_model = BertForSequenceClassification.from_pretrained("CPDT/aspect-sentiment-clf")
        text_tokenizer = AutoTokenizer.from_pretrained("CPDT/aspect-sentiment-clf")
        device = "cpu"

        print(f"Model loaded successfully on {device}")
        return aspect_extractor,aspect_sentiment,sentiment_model,text_tokenizer, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run specify path of the model")
        raise


try:
    aspect_extractor,aspect_sentiment,sentiment_model,text_tokenizer, device = load_model_and_tokenizer()
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    aspect_extractor,aspect_sentiment,sentiment_model,text_tokenizer, device = None, None, None , None , None 


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
    if sentiment_model is None or text_tokenizer is None:
        raise RuntimeError("Model not loaded. Please Give path or Download model first.")

    language = detect_language(text)
    cleaned_text = preprocess(text)

    if language != "en":
        translated = translator.translate(text, dest="en")
        text = translated.text

    inputs = text_tokenizer(text, truncation=True, padding=True, max_length=128,return_tensors="pt")

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)

    raw_aspect_results  = absa_pipeline(cleaned_text)

    predictions = []
    for label, prob in zip(LABELS, probs.cpu().tolist()):
        predictions.append({
            "label": label,
            "score": round(prob, 4)
        })


    formatted_aspects = []
    for item in raw_aspect_results:
        aspect_dict = {
            "aspect": item["aspect"],
            "sentiment": item["sentiment"].lower(),
            "confidence": round(item["confidence"], 4)
        }
        formatted_aspects.append(aspect_dict)


    predictions.sort(key=lambda x: x["score"], reverse=True)

    return {
        "predictions": predictions,
        "top_prediction": predictions[0],
        "aspects_list": formatted_aspects
    }


def get_top_prediction(text: str) -> Dict[str, any]:
    """Get only the top prediction"""
    results = predict_sentiment(text)
    return results[0]


def absa_pipeline(sentence):
    results = []

    # Step 1: Extract aspects
    extracted = aspect_extractor(sentence)
    print("Extracted entities:", extracted)

    # Keep only ASP entities
    aspects = [item["word"] for item in extracted if item["entity_group"] == "ASP"]
    print("Filtered ASP aspects:", aspects)  

    # Remove duplicates
    aspects = list(set(aspects))

    # Step 2: Predict sentiment for each aspect
    for aspect in aspects:
        sentiment_result = aspect_sentiment(inputs=aspect, text_pair=sentence)[0]
        results.append(
            {
                "aspect": aspect,
                "sentiment": sentiment_result["label"],
                "confidence": round(sentiment_result["score"], 8),
            }
        )
    return results
