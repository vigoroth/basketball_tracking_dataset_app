import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Adjust these imports if your code structure is different
from routes import api_router
from utils import ensure_directories

# Example: If you want to ensure certain directories exist at startup,
# uncomment or modify these lines:
# DATASET_BASE_PATH = "/path/to/dataset"
# os.makedirs(DATASET_BASE_PATH, exist_ok=True)
# ensure_directories(DATASET_BASE_PATH)

# Create the FastAPI app
app = FastAPI(
    title="Basketball Tracking API",
    description="An API for court, player, and ball detection/tracking.",
    version="1.0.0"
)


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes from routes.py
app.include_router(api_router)

# Entry point
if __name__ == "__main__":
    # Run with: python main.py
    # Or: uvicorn main:app --host 0.0.0.0 --port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
