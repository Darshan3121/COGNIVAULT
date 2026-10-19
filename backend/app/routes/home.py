from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.get("/")
def home():
    return {
        "message": "Welcome to AKOS AI"
    }