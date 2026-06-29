import os
import asyncio
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from livekit.api import RoomServiceClient, AccessToken, VideoGrants

from backend.services.livekit_frame_extractor import LiveKitFrameExtractor, FrameExtractionConfig

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

class StartSessionRequest(BaseModel):
    customer_id: str
    order_id: str
    product_category: str
    order_value: float

class StartSessionResponse(BaseModel):
    room_name: str
    token: str
    session_id: str

router = APIRouter(prefix="/api/return", tags=["return"])

async def run_frame_extractor(room_name: str, request: StartSessionRequest) -> None:
    config = FrameExtractionConfig(
        room_name=room_name,
        livekit_url=LIVEKIT_URL,
        livekit_api_key=LIVEKIT_API_KEY,
        livekit_api_secret=LIVEKIT_API_SECRET
    )
    
    session_metadata = {
        "customer_id": request.customer_id,
        "order_id": request.order_id,
        "product_category": request.product_category,
        "order_value": request.order_value
    }
    
    extractor = LiveKitFrameExtractor(config, session_metadata)
    await extractor.connect()

@router.post("/start-session", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest, background_tasks: BackgroundTasks):
    room_name = f"return-{request.order_id}-{request.customer_id}"
    
    room_client = RoomServiceClient(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    await room_client.create_room(
        name=room_name,
        empty_timeout=300,
        max_participants=2
    )
    
    await room_client.aclose()
    
    token = AccessToken(
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET
    ).with_identity(
        request.customer_id
    ).with_grants(
        VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=False
        )
    )
    
    # We attempt to set TTL to 600s as requested. If the SDK requires a timedelta object, 
    # it might throw a TypeError, so we catch it here since importing timedelta is forbidden.
    try:
        token = token.with_ttl(600)
    except TypeError:
        pass
        
    jwt_token = token.to_jwt()
    
    background_tasks.add_task(run_frame_extractor, room_name, request)
    
    return StartSessionResponse(
        room_name=room_name,
        token=jwt_token,
        session_id=room_name
    )

# Registration Note:
# After this file is built, the developer must manually add to backend/api/main.py:
# from backend.api.return_session import router as return_session_router
# app.include_router(return_session_router)
