from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from predictor import Predictor

import io
import os
import uuid
import time
import psycopg


app = FastAPI(title="SketchMind API")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

predictor = Predictor()


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL configured:", bool(DATABASE_URL))
print("DATABASE_URL starts with:", DATABASE_URL[:20] if DATABASE_URL else "NONE")

def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL environment variable is not configured."
        )

    return psycopg.connect(DATABASE_URL)


def init_database():

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS leaderboard (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    correct INTEGER NOT NULL,
                    rounds INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    best_streak INTEGER NOT NULL,
                    played_at BIGINT NOT NULL
                )
                """
            )

        connection.commit()

    finally:
        connection.close()


@app.on_event("startup")
def startup():

    try:
        init_database()
        print("Leaderboard database initialized successfully.")

    except Exception as e:
        print(f"Database initialization failed: {e}")


# --------------------------------------------------
# LEADERBOARD MODEL
# --------------------------------------------------

class ScoreSubmission(BaseModel):

    name: str = Field(min_length=1, max_length=30)

    score: int = Field(ge=0)

    correct: int = Field(ge=0)

    rounds: int = Field(gt=0)

    difficulty: str

    bestStreak: int = Field(ge=0)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "SketchMind API"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# ML PREDICTION
# --------------------------------------------------

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


# --------------------------------------------------
# SAVE SCORE
# --------------------------------------------------

@app.post("/scores")
def save_score(entry: ScoreSubmission):

    name = entry.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name cannot be empty."
        )

    if entry.correct > entry.rounds:
        raise HTTPException(
            status_code=400,
            detail="Correct answers cannot exceed rounds."
        )

    if entry.bestStreak > entry.correct:
        raise HTTPException(
            status_code=400,
            detail="Best streak cannot exceed correct answers."
        )

    score_id = str(uuid.uuid4())

    played_at = int(time.time() * 1000)

    connection = get_connection()

    try:

        with connection.cursor(
            row_factory=psycopg.rows.dict_row
        ) as cursor:

            cursor.execute(
                """
                INSERT INTO leaderboard
                (
                    id,
                    name,
                    score,
                    correct,
                    rounds,
                    difficulty,
                    best_streak,
                    played_at
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                RETURNING
                    id,
                    name,
                    score,
                    correct,
                    rounds,
                    difficulty,
                    best_streak AS "bestStreak",
                    played_at AS "playedAt"
                """,
                (
                    score_id,
                    name,
                    entry.score,
                    entry.correct,
                    entry.rounds,
                    entry.difficulty,
                    entry.bestStreak,
                    played_at,
                )
            )

            result = cursor.fetchone()

        connection.commit()

        return dict(result)

    except Exception as e:

        connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Could not save score: {str(e)}"
        )

    finally:

        connection.close()


# --------------------------------------------------
# GET LEADERBOARD
# --------------------------------------------------

@app.get("/leaderboard")
def get_leaderboard():

    connection = get_connection()

    try:

        with connection.cursor(
            row_factory = psycopg.rows.dict_row
        ) as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    score,
                    correct,
                    rounds,
                    difficulty,
                    best_streak AS "bestStreak",
                    played_at AS "playedAt"
                FROM leaderboard
                ORDER BY
                    score DESC,
                    played_at DESC
                LIMIT 100
                """
            )

            results = cursor.fetchall()

        return [dict(row) for row in results]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not load leaderboard: {str(e)}"
        )

    finally:

        connection.close()
