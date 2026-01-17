from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from fastapi import FastAPI , Request , Form
from fastapi.responses import HTMLResponse, JSONResponse , RedirectResponse
from model import predict_sentiment , SentimentResponse , SentimentRequest
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



app = FastAPI(title="Sentiment Analysis Api",version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
async def inputs(request:Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...)
):
    try:
        result = await predict_sentiment(text)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "text": text,
                "predictions": result["predictions"],
                "top_prediction": result["top_prediction"],
                "error": None
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "text": text,
                "predictions": None,
                "top_prediction": None,
                "error": str(e)
            }
        )

@app.post("/api/predict", response_model=SentimentResponse)
async def predict_api(request: SentimentRequest):
    try:
        predictions = predict_sentiment(request.text)
        
        return SentimentResponse(
            predictions=[
                SentimentResult(label=p["label"], score=p["score"])
                for p in predictions
            ],
            top_prediction=SentimentResult(
                label=predictions[0]["label"],
                score=predictions[0]["score"]
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/api/labels")
async def get_labels():
    """Get available sentiment labels"""
    from model import LABELS
    return {"labels": LABELS}