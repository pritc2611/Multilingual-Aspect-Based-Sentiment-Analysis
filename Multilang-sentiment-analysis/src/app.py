from fastapi import FastAPI , Request , Form , HTTPException
from fastapi.responses import HTMLResponse, JSONResponse , RedirectResponse
from model import predict_sentiment , SentimentResponse , SentimentRequest , SentimentResult , AspectResult
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
                "aspects": result["aspects_list"],
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
        # result is now the dictionary returned by predict_sentiment
        result = await predict_sentiment(request.text)
        
        return SentimentResponse(
            predictions=[
                SentimentResult(label=p["label"], score=p["score"]) 
                for p in result["predictions"]
            ],
            top_prediction=SentimentResult(
                label=result["top_prediction"]["label"], 
                score=result["top_prediction"]["score"]
            ),
            aspects=[
                AspectResult(
                    aspect=a["aspect"],
                    sentiment=a["sentiment"],
                    confidence=a["confidence"]
                ) for a in result["aspects_list"]
            ]
        )
    except Exception as e:
        print(f"Detailed Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        

@app.get("/api/health")
async def health_check():
    from model import sentiment_model

    return {
        "status": "healthy" if sentiment_model is not None else "unhealthy",
        "model_loaded": sentiment_model is not None
    }


@app.get("/api/labels")
async def get_labels():
    """Get available sentiment labels"""
    from model import LABELS
    return {"labels": LABELS}


