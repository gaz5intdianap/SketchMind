from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from predictor import Predictor
import io

app = FastAPI(title="SketchMind API")

# Allow the Lovable/Vercel frontend to communicate with Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when the API starts
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

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    label, confidence = predictor.predict(image)

    return {
        "prediction": label,
        "confidence": confidence
    }