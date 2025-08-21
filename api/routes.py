from fastapi import APIRouter

router = APIRouter()

@router.get("/chat")
async def chat_endpoint():
    return {"ok": True}
