from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys

# Force UTF-8 encoding for standard output to prevent Windows console crashes on Unicode characters (like emojis or special PDF characters in Analytics logs)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from app.routes.home import router as home_router
from app.routes.chat import router as chat_router
from app.routes.upload import router as upload_router
from app.routes.analytics import router as analytics_router


app = FastAPI(
    title="Cognivault API",
    description="Intelligent Document Assistant",
    version="1.0.0"
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

app.include_router(home_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(analytics_router)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Welcome to Cognivault"
    }