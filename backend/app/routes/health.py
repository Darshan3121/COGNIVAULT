from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AKOS AI Backend"
    }