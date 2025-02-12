from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import asyncio

# Create the FastAPI app instance
app = FastAPI(title="Sentiment Analysis API")

# Load the sentiment analysis pipeline with the DistilBERT model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Define the request body schema
class SentimentRequest(BaseModel):
    sentence: str

# Define an endpoint to get sentiment analysis
@app.post("/sentiment", summary="Get Sentiment of a Text")
async def get_sentiment(request: SentimentRequest):
    # Offload the blocking inference to a separate thread
    result = await asyncio.to_thread(sentiment_model, request.sentence)
    return result

# Optional health check endpoint
@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "ok"}

# Run the server with Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=4)