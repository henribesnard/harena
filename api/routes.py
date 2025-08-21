from fastapi import APIRouter

from conversation_service.api.routes import router as conversation_router


router = APIRouter()
router.include_router(conversation_router, prefix="/conversation")
