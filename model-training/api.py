from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from predictor import Predictor
import io

app = FastAPI(title="SketchMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SketchMind API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        label, confidence = predictor.predict(image)

        return {
            "prediction": label,
            "confidence": float(confidence)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
