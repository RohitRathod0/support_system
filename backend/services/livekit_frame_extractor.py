import os
import asyncio
import base64
import io
from dataclasses import dataclass
from PIL import Image
from livekit import rtc, api

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

@dataclass
class FrameExtractionConfig:
    room_name: str
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    frame_interval_seconds: int = 3
    max_frames: int = 5

class LiveKitFrameExtractor:
    def __init__(self, config: FrameExtractionConfig, session_metadata: dict):
        self.config = config
        self.session_metadata = session_metadata
        self.frames: list[str] = []
        self.room = None
        self._extraction_done = False

    async def connect(self) -> None:
        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track_subscribed)
        
        token = api.AccessToken(
            self.config.livekit_api_key,
            self.config.livekit_api_secret
        ).with_identity("support-agent-observer").with_name("support-agent-observer").with_grants(
            api.VideoGrants(room_join=True, room=self.config.room_name)
        ).to_jwt()

        await self.room.connect(self.config.livekit_url, token)

    async def _on_track_subscribed(self, track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(self._extract_frames(track))

    async def _extract_frames(self, track) -> None:
        video_stream = rtc.VideoStream(track)
        
        async for frame_event in video_stream:
            if self._extraction_done:
                break
                
            frame = frame_event.frame
            
            video_frame_rgba = frame.convert(rtc.VideoBufferType.RGBA)
            image = Image.frombytes("RGBA", (video_frame_rgba.width, video_frame_rgba.height), video_frame_rgba.data)
            
            buffered = io.BytesIO()
            image.convert("RGB").save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            self.frames.append(img_str)
            
            if len(self.frames) >= self.config.max_frames:
                self._extraction_done = True
                await self._trigger_agent()
                if self.room:
                    await self.room.disconnect()
                return
                
            await asyncio.sleep(self.config.frame_interval_seconds)

    async def _trigger_agent(self) -> None:
        from backend.agents.video_return_agent import video_return_graph
        
        initial_state = {
            "session_id": self.config.room_name,
            "customer_id": self.session_metadata["customer_id"],
            "order_id": self.session_metadata["order_id"],
            "product_category": self.session_metadata["product_category"],
            "order_value": self.session_metadata["order_value"],
            "video_frames": self.frames,
            "vision_output": {},
            "policy_context": "",
            "classification_tier": "",
            "confidence_score": 0.0,
            "decision_reason": "",
            "final_action": ""
        }
        
        self.agent_result = await video_return_graph.ainvoke(initial_state)
        
        final_action = self.agent_result.get("final_action", "")
        decision_reason = self.agent_result.get("decision_reason", "")
        print(f"Agent Final Action: {final_action}")
        print(f"Agent Decision Reason: {decision_reason}")
