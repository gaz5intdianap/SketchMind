from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from predictor import Predictor
import io

app = FastAPI(title="SketchMind API")

# Allow the frontend to communicate with Railway
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

    try:
        # Read uploaded image
        image_bytes = await file.read()

        # Open image
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Handle transparency consistently
        if image.mode == "RGBA":
            background = Image.new(
                "RGB",
                image.size,
                (0, 0, 0)
            )

            background.paste(
                image,
                mask=image.getchannel("A")
            )

            image = background

        elif image.mode != "RGB":
            image = image.convert("RGB")

        # IMPORTANT:
        # Predictor handles grayscale, cropping,
        # centering and resizing.
        label, confidence = predictor.predict(image)

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )