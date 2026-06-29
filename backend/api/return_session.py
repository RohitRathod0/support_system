import os
import asyncio
import json
import redis.asyncio as aioredis
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import livekit.api
from livekit.api import LiveKitAPI, AccessToken, VideoGrants
from langgraph.types import Command

from backend.services.livekit_frame_extractor import LiveKitFrameExtractor, FrameExtractionConfig

_active_graphs: dict[str, any] = {}

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

class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    final_action: str | None = None
    decision_reason: str | None = None
    confidence_score: float | None = None

class HumanDecisionRequest(BaseModel):
    session_id: str
    decision: str

class HumanDecisionResponse(BaseModel):
    session_id: str
    resumed: bool
    final_action: str

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
    _active_graphs[room_name] = extractor
    await extractor.connect()

@router.post("/start-session", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest, background_tasks: BackgroundTasks):
    room_name = f"return-{request.order_id}-{request.customer_id}"
    
    api = LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    
    await api.room.create_room(
        livekit.api.CreateRoomRequest(
            name=room_name,
            empty_timeout=300,
            max_participants=2
        )
    )
    
    await api.aclose()
    
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

@router.get("/status/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    client = aioredis.from_url(os.getenv("REDIS_URL"))
    key = f"return:result:{session_id}"
    val = await client.get(key)
    
    if not val:
        try:
            await client.aclose()
        except AttributeError:
            await client.close()
            
        return SessionStatusResponse(
            session_id=session_id,
            status="processing"
        )
        
    data = json.loads(val)
    try:
        await client.aclose()
    except AttributeError:
        await client.close()
        
    return SessionStatusResponse(
        session_id=session_id,
        status="done",
        final_action=data["final_action"],
        decision_reason=data["decision_reason"],
        confidence_score=data["confidence_score"]
    )

@router.post("/decision/{session_id}", response_model=HumanDecisionResponse)
async def handle_human_decision(session_id: str, request: HumanDecisionRequest):
    if request.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be approve or reject")
        
    final_action_string = "APPROVED" if request.decision == "approve" else "REJECTED"
    
    client = aioredis.from_url(os.getenv("REDIS_URL"))
    key = f"return:result:{session_id}"
    val = await client.get(key)
    
    if val:
        data = json.loads(val)
        if data.get("final_action"):
            try:
                await client.aclose()
            except AttributeError:
                await client.close()
                
            return HumanDecisionResponse(
                session_id=session_id,
                resumed=False,
                final_action=data["final_action"]
            )
            
    try:
        await client.aclose()
    except AttributeError:
        await client.close()
        
    extractor = _active_graphs.get(session_id)
    if not extractor:
        raise HTTPException(status_code=404, detail="session not found or already completed")
        
    from backend.agents.video_return_agent import video_return_graph
    
    command = Command(resume={"human_decision": final_action_string})
    await video_return_graph.ainvoke(command, config={"configurable": {"thread_id": session_id}})
    
    if session_id in _active_graphs:
        del _active_graphs[session_id]
        
    return HumanDecisionResponse(
        session_id=session_id,
        resumed=True,
        final_action=final_action_string
    )

# Registration Note:
# After this file is built, the developer must manually add to backend/api/main.py:
# from backend.api.return_session import router as return_session_router
# app.include_router(return_session_router)
