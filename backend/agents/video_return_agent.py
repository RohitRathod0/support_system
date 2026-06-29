import os
import json
import redis.asyncio as aioredis
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
import google.generativeai as genai
import chromadb

class ReturnAgentState(TypedDict):
    session_id: str
    customer_id: str
    order_id: str
    product_category: str
    order_value: float
    video_frames: List[str]
    vision_output: Dict[str, Any]
    policy_context: str
    classification_tier: str
    confidence_score: float
    decision_reason: str
    final_action: str

def analyze_video(state: ReturnAgentState) -> ReturnAgentState:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    Please analyze these video frames and provide a structured JSON output with these exact keys:
    - defect_detected (bool)
    - confidence (float 0-1)
    - defect_type (str)
    - defect_description (str)
    - product_matches_order (bool)
    - product_condition (str)
    - fraud_indicators (list of strings)
    - recommended_action (str)
    """
    
    parts = []
    for frame in state.get("video_frames", []):
        parts.append({"mime_type": "image/jpeg", "data": frame})
    
    contents = [prompt] + parts
    
    response = model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    
    try:
        vision_output = json.loads(response.text)
    except json.JSONDecodeError:
        vision_output = {
            "defect_detected": False,
            "confidence": 0.0,
            "defect_type": "",
            "defect_description": "Failed to parse JSON response",
            "product_matches_order": False,
            "product_condition": "unknown",
            "fraud_indicators": [],
            "recommended_action": "manual review"
        }
        
    return {"vision_output": vision_output}

def fetch_policy(state: ReturnAgentState) -> ReturnAgentState:
    db_path = os.getenv("CHROMA_DB_PATH", "./vector_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("company_policies")
    
    query = f"return policy for {state.get('product_category', '')}"
    results = collection.query(query_texts=[query], n_results=2)
    
    policy_context = ""
    if results and results.get("documents") and len(results["documents"]) > 0:
        policy_context = "\n".join(results["documents"][0])
        
    return {"policy_context": policy_context}

def classify_return(state: ReturnAgentState) -> ReturnAgentState:
    vision = state.get("vision_output", {})
    category = state.get("product_category", "")
    order_value = state.get("order_value", 0.0)
    
    confidence = vision.get("confidence", 0.0)
    defect_detected = vision.get("defect_detected", False)
    product_matches_order = vision.get("product_matches_order", False)
    fraud_indicators = vision.get("fraud_indicators", [])
    defect_type = vision.get("defect_type", "")
    
    # Priority 1: HUMAN_QUEUE
    is_human_queue = False
    if category in ["electronics", "appliances", "jewelry", "furniture"]:
        is_human_queue = True
    elif order_value >= 5000:
        is_human_queue = True
    elif 0.50 <= confidence <= 0.79:
        is_human_queue = True
    elif defect_type == "functional":
        is_human_queue = True
    elif len(fraud_indicators) > 0:
        is_human_queue = True
        
    if is_human_queue:
        return {
            "classification_tier": "human_queue",
            "confidence_score": confidence,
            "decision_reason": "Flags raised requiring human review."
        }
        
    # Priority 2: AUTO-APPROVE
    is_auto_approve = False
    if (confidence >= 0.80 and 
        order_value < 2000 and 
        category in ["clothing", "footwear", "accessories", "books", "toys"] and 
        defect_detected is True and 
        product_matches_order is True and 
        len(fraud_indicators) == 0):
        is_auto_approve = True
        
    if is_auto_approve:
        return {
            "classification_tier": "auto_approve",
            "confidence_score": confidence,
            "decision_reason": "Met all criteria for auto-approval."
        }
        
    # Priority 3: AUTO-REJECT
    is_auto_reject = False
    if defect_detected is False and confidence >= 0.75:
        is_auto_reject = True
    elif product_matches_order is False:
        is_auto_reject = True
    else:
        # Does not meet auto_approve and does not meet human_queue
        is_auto_reject = True
        
    if is_auto_reject:
        return {
            "classification_tier": "auto_reject",
            "confidence_score": confidence,
            "decision_reason": "Failed requirements for return approval."
        }
    
    return {}

async def persist_result(session_id: str, final_action: str, decision_reason: str, confidence_score: float) -> None:
    client = aioredis.from_url(os.getenv("REDIS_URL"))
    data = {
        "session_id": session_id,
        "final_action": final_action,
        "decision_reason": decision_reason,
        "confidence_score": confidence_score
    }
    key = f"return:result:{session_id}"
    await client.set(key, json.dumps(data), ex=3600)
    try:
        await client.aclose()
    except AttributeError:
        await client.close()

async def route_decision(state: ReturnAgentState) -> ReturnAgentState:
    tier = state.get("classification_tier")
    
    if tier == "auto_approve":
        state["final_action"] = "APPROVED"
        await persist_result(
            session_id=state["session_id"],
            final_action=state["final_action"],
            decision_reason=state["decision_reason"],
            confidence_score=state["confidence_score"]
        )
        return {"final_action": "APPROVED"}
    elif tier == "auto_reject":
        state["final_action"] = "REJECTED"
        await persist_result(
            session_id=state["session_id"],
            final_action=state["final_action"],
            decision_reason=state["decision_reason"],
            confidence_score=state["confidence_score"]
        )
        return {"final_action": "REJECTED"}
    elif tier == "human_queue":
        vision = state.get("vision_output", {})
        payload = {
            "session_id": state.get("session_id"),
            "customer_id": state.get("customer_id"),
            "order_id": state.get("order_id"),
            "vision_output": vision,
            "decision_reason": state.get("decision_reason"),
            "confidence_score": state.get("confidence_score"),
            "recommended_action": vision.get("recommended_action")
        }
        human_decision = interrupt(payload)
        
        final_action = ""
        if isinstance(human_decision, str):
            if human_decision.lower() == "approve":
                final_action = "APPROVED"
            elif human_decision.lower() == "reject":
                final_action = "REJECTED"
                
        state["final_action"] = final_action
        await persist_result(
            session_id=state["session_id"],
            final_action=state["final_action"],
            decision_reason=state["decision_reason"],
            confidence_score=state["confidence_score"]
        )
                
        return {"final_action": final_action}
    
    return {}

builder = StateGraph(ReturnAgentState)

builder.add_node("analyze_video", analyze_video)
builder.add_node("fetch_policy", fetch_policy)
builder.add_node("classify_return", classify_return)
builder.add_node("route_decision", route_decision)

builder.add_edge(START, "analyze_video")
builder.add_edge("analyze_video", "fetch_policy")
builder.add_edge("fetch_policy", "classify_return")
builder.add_edge("classify_return", "route_decision")
builder.add_edge("route_decision", END)

video_return_graph = builder.compile()
