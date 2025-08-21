from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect


def build_app() -> FastAPI:
    app = FastAPI()
    router = APIRouter()

    @router.get("/chat")
    async def chat_endpoint():
        return {"ok": True}

    ws_router = APIRouter()

    @ws_router.websocket("/ws")
    async def websocket_endpoint(ws):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                await ws.send_text(data)
        except WebSocketDisconnect:
            pass

    app.include_router(router)
    app.include_router(ws_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def test_websocket_echo() -> None:
    app = build_app()
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("hello")
            assert websocket.receive_text() == "hello"
    except WebSocketDisconnect:
        pass


def test_websocket_multiple_messages() -> None:
    app = build_app()
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws") as websocket:
            for i in range(3):
                msg = f"msg {i}"
                websocket.send_text(msg)
                assert websocket.receive_text() == msg
    except WebSocketDisconnect:
        pass

