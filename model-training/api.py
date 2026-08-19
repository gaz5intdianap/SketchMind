from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from predictor import Predictor

import io
import os
import uuid
import time
import redis


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
# REDIS DATABASE
# --------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL")

print("REDIS_URL configured:", bool(REDIS_URL))

if not REDIS_URL:
    raise RuntimeError(
        "REDIS_URL environment variable is not configured."
    )

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True
)

LEADERBOARD_KEY = "sketchmind:leaderboard"


def get_redis():
    return redis_client


@app.on_event("startup")
def startup():
    print("API startup successful.")


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
# DATABASE TEST
# --------------------------------------------------

@app.get("/db-test")
def db_test():

    try:

        print("Testing Redis connection...")

        client = get_redis()

        result = client.ping()

        print("Redis connection successful.")

        return {
            "status": "success",
            "redis": result
        }

    except Exception as e:

        print(f"Redis error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Redis connection failed: {str(e)}"
        )
        
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

    client = get_redis()

    try:

        score_data = {
            "id": score_id,
            "name": name,
            "score": entry.score,
            "correct": entry.correct,
            "rounds": entry.rounds,
            "difficulty": entry.difficulty,
            "bestStreak": entry.bestStreak,
            "playedAt": played_at
        }

        # Store complete score information
        client.hset(
            f"score:{score_id}",
            mapping=score_data
        )

        # Add score to leaderboard
        client.zadd(
            LEADERBOARD_KEY,
            {
                score_id: entry.score
            }
        )

        return score_data

    except Exception as e:

        print(f"Redis error while saving score: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Could not save score: {str(e)}"
        )

# --------------------------------------------------
# GET LEADERBOARD
# --------------------------------------------------

@app.get("/leaderboard")
def get_leaderboard():

    client = get_redis()

    try:

        # Get top 100 scores, highest first
        score_ids = client.zrevrange(
            LEADERBOARD_KEY,
            0,
            99
        )

        leaderboard = []

        for score_id in score_ids:

            score_data = client.hgetall(
                f"score:{score_id}"
            )

            if score_data:

                score_data["score"] = int(
                    score_data["score"]
                )

                score_data["correct"] = int(
                    score_data["correct"]
                )

                score_data["rounds"] = int(
                    score_data["rounds"]
                )

                score_data["bestStreak"] = int(
                    score_data["bestStreak"]
                )

                score_data["playedAt"] = int(
                    score_data["playedAt"]
                )

                leaderboard.append(score_data)

        return leaderboard

    except Exception as e:

        print(f"Redis error while loading leaderboard: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Could not load leaderboard: {str(e)}"
        )

# --------------------------------------------------
# ADMIN STATS - ALL SCORES
# --------------------------------------------------

@app.get("/admin/stats")
def get_admin_stats():

    client = get_redis()

    try:

        # Get ALL score IDs from Redis
        score_ids = client.zrevrange(
            LEADERBOARD_KEY,
            0,
            -1
        )

        all_scores = []

        for score_id in score_ids:

            score_data = client.hgetall(
                f"score:{score_id}"
            )

            if score_data:

                score_data["score"] = int(
                    score_data["score"]
                )

                score_data["correct"] = int(
                    score_data["correct"]
                )

                score_data["rounds"] = int(
                    score_data["rounds"]
                )

                score_data["bestStreak"] = int(
                    score_data["bestStreak"]
                )

                score_data["playedAt"] = int(
                    score_data["playedAt"]
                )

                all_scores.append(score_data)

        # --------------------------------------------------
        # OVERALL STATISTICS
        # --------------------------------------------------

        total_games = len(all_scores)

        if total_games > 0:

            total_score = sum(
                item["score"]
                for item in all_scores
            )

            total_correct = sum(
                item["correct"]
                for item in all_scores
            )

            total_rounds = sum(
                item["rounds"]
                for item in all_scores
            )

            highest_score = max(
                item["score"]
                for item in all_scores
            )

            best_streak = max(
                item["bestStreak"]
                for item in all_scores
            )

            average_score = (
                total_score / total_games
            )

            average_accuracy = (
                (total_correct / total_rounds) * 100
                if total_rounds > 0
                else 0
            )

        else:

            highest_score = 0
            best_streak = 0
            average_score = 0
            average_accuracy = 0

        return {
            "totalGames": total_games,
            "highestScore": highest_score,
            "averageScore": round(
                average_score,
                2
            ),
            "averageAccuracy": round(
                average_accuracy,
                2
            ),
            "bestStreak": best_streak,
            "allScores": all_scores
        }

    except Exception as e:

        print(
            f"Redis error while loading admin stats: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not load admin stats: {str(e)}"
        )
