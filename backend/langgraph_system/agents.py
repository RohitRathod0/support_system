"""
LangGraph Agent Nodes — Customer Support System
All 11 CrewAI agents reimplemented as async LangGraph nodes.

Parallel execution:  kb_retriever + policy_retriever + web_search run simultaneously
                     escalation + cx_optimizer run simultaneously post-persist
QA retry loop:       qa_review → generate_solution (max 2 attempts)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from .state import SupportState

logger = logging.getLogger(__name__)

# ─── LLM ─────────────────────────────────────────────────────────────────────
def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def _call_llm(system: str, human: str) -> str:
    """Synchronous LLM call with error handling."""
    llm = _get_llm()
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"[LLM ERROR: {e}]"

async def _acall_llm(system: str, human: str) -> str:
    """Async LLM call."""
    llm = _get_llm()
    try:
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=human)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Async LLM call failed: {e}")
        return f"[LLM ERROR: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 · TICKET CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
def detect_resolution(message: str) -> bool:
    """Returns True if the customer message signals the issue is resolved."""
    signals = [
        "resolved", "satisfied", "thank you", "thanks", "problem solved",
        "issue fixed", "all good", "that's all", "no more questions",
        "query is resolved", "query resolved", "issue resolved",
        "happy with", "that helps", "got it thank", "appreciate your help",
        "great help", "good help", "my problem is solved",
    ]
    msg_lower = message.lower()
    return any(s in msg_lower for s in signals)


def check_for_defect_language(message: str) -> bool:
    keywords = [
        "hole", "torn", "tear", "rip", "damaged", "damage", "defective",
        "defect", "broken", "crack", "cracked", "scratch", "scratched",
        "stain", "stained", "dent", "dented", "shattered", "burnt",
        "physical damage", "not working", "stopped working", "fell apart"
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in keywords)

def classify_ticket_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: ticket_classifier agent (agents.yaml)
    Task:    ticket_classification_task (tasks.yaml)
    """
    start = time.time()
    query = state["customer_query"]
    
    defect_detected = check_for_defect_language(query)
    resolution_detected = detect_resolution(query)

    system = """You are a Senior Customer Support Ticket Classifier with 10+ years of experience.
Analyze the customer query and return a JSON object with EXACTLY these fields:
{
  "urgency_level": "Critical|High|Medium|Low",
  "issue_category": "Technical|Billing|Account|Product|General",
  "sentiment": "Satisfied|Neutral|Frustrated|Angry",
  "complexity": "Simple|Moderate|Complex|Expert",
  "special_flags": ["list of: VIP|Legal|Compliance|Escalation|Privacy"],
  "primary_issue": "one-sentence summary",
  "secondary_issues": ["list of related concerns"],
  "recommended_team": "Tier1|Tier2|Specialist|Management",
  "reasoning": "brief explanation of classification"
}
Return ONLY valid JSON, no markdown, no extra text."""

    human = f"Customer Query: {query}\nCurrent Year: {state.get('current_year', '2025')}"
    raw = _call_llm(system, human)

    try:
        data = json.loads(raw)
    except Exception:
        # Graceful parse fallback
        data = {
            "urgency_level": "Medium",
            "issue_category": "General",
            "sentiment": "Neutral",
            "complexity": "Moderate",
            "special_flags": [],
            "primary_issue": query[:100],
            "secondary_issues": [],
            "recommended_team": "Tier1",
            "reasoning": "Classification parsing failed, using defaults",
        }

    timings = dict(state.get("node_timings") or {})
    timings["classify_ticket"] = round(time.time() - start, 2)

    return {
        "urgency_level": data.get("urgency_level", "Medium"),
        "issue_category": data.get("issue_category", "General"),
        "sentiment": data.get("sentiment", "Neutral"),
        "complexity": data.get("complexity", "Moderate"),
        "special_flags": data.get("special_flags", []),
        "ticket_classification": data,
        "defect_language_detected": defect_detected,
        "resolution_detected": resolution_detected,
        "node_timings": timings,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1.5 · CONTRADICTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
def detect_contradictions_node(state: SupportState) -> Dict[str, Any]:
    start = time.time()
    current_msg = state["customer_query"].lower()
    history = state.get("conversation_history", [])
    
    damage_keywords = ["hole", "torn", "damaged", "defective", "broken", "scratch", "crack", "stain"]
    has_damage = any(kw in current_msg for kw in damage_keywords)
    
    contradiction_detected = False
    contradiction_type = ""
    contradiction_message = ""
    
    if has_damage:
        for turn in history:
            prev_msg = turn.get("user", "").lower()
            
            # Pattern 1
            if any(kw in prev_msg for kw in ["not arrived", "hasn't arrived", "didn't arrive", "never received", "not received"]):
                contradiction_detected = True
                contradiction_type = "arrival_vs_damage"
                contradiction_message = "You mentioned earlier that your order hasn't arrived, but now you're describing physical damage to the product. Could you clarify — did the order actually arrive? If it did arrive and is damaged, that changes how we can help you."
                break
                
            # Pattern 2
            if any(kw in prev_msg for kw in ["wrong item", "wrong product", "not what i ordered", "different item"]):
                contradiction_detected = True
                contradiction_type = "wrong_item_vs_damage"
                contradiction_message = "You mentioned receiving the wrong item, but now you're describing damage to it. Could you clarify which issue we should address first — the wrong item or the damage?"
                break
                
    timings = dict(state.get("node_timings") or {})
    timings["detect_contradictions"] = round(time.time() - start, 2)
    
    result = {
        "contradiction_detected": contradiction_detected,
        "contradiction_type": contradiction_type,
        "contradiction_message": contradiction_message,
        "node_timings": timings,
    }
    
    if contradiction_detected:
        result["personalized_response"] = contradiction_message
        
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 · SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
def manage_session_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: session_manager agent
    Task:    manage_user_session
    """
    start = time.time()
    user_id = state.get("user_id", "anonymous")
    history = state.get("conversation_history", [])
    classification = state.get("ticket_classification", {})

    system = """You are a User Session and Context Manager. Given the user's classification and history,
return a JSON object with EXACTLY these fields:
{
  "session_type": "New|Continuing|Escalated",
  "account_status": "Active|Suspended|Premium|Trial|Unknown",
  "communication_style": "Formal|Casual|Technical|Simple",
  "expertise_level": "Beginner|Intermediate|Expert",
  "response_format": "DetailedSteps|QuickSummary|BulletPoints",
  "previous_issues_summary": "brief summary or none",
  "personalization_notes": "what to personalize in the response",
  "context_confidence": 0.8
}
Return ONLY valid JSON."""

    human = f"""User ID: {user_id}
Classification: {json.dumps(classification)}
Conversation history turns: {len(history)}
Recent history: {json.dumps(history[-3:] if history else [])}"""

    raw = _call_llm(system, human)
    try:
        session_data = json.loads(raw)
    except Exception:
        session_data = {
            "session_type": "New" if not history else "Continuing",
            "account_status": "Active",
            "communication_style": "Professional",
            "expertise_level": "Intermediate",
            "response_format": "DetailedSteps",
            "personalization_notes": "Standard professional response",
            "context_confidence": 0.5,
        }

    timings = dict(state.get("node_timings") or {})
    timings["manage_session"] = round(time.time() - start, 2)

    return {
        "session_context": session_data,
        "personalization": {
            "style": session_data.get("communication_style", "Professional"),
            "expertise": session_data.get("expertise_level", "Intermediate"),
            "format": session_data.get("response_format", "DetailedSteps"),
        },
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 · PARALLEL RETRIEVAL (KB + Policy + Web run simultaneously)
# ═══════════════════════════════════════════════════════════════════════════════
async def _retrieve_kb(state: SupportState, kb_service) -> List[Dict]:
    """
    Maps to: kb_retriever agent
    Task:    kb_retrieval_task
    """
    query = state["customer_query"]
    category = state.get("issue_category", "General")

    # ChromaDB search
    results = kb_service.search(query, n_results=5) if kb_service else []

    # Also get conversation context
    conv_ctx = (
        kb_service.get_conversation_context(query, state.get("user_id", "anon"), n_results=2)
        if kb_service else []
    )

    if not results:
        # LLM fallback for KB
        system = "You are an Internal Knowledge Base Specialist. Provide relevant support knowledge."
        human = f"Query: {query}\nCategory: {category}\nProvide 2-3 relevant knowledge points."
        content = await _acall_llm(system, human)
        results = [{"content": content, "score": 0.6, "source": "llm_kb_fallback"}]

    return results + [{"content": c["content"], "score": 0.5, "source": "conversation_history"}
                      for c in conv_ctx]


async def _retrieve_policies(state: SupportState, policy_service) -> List[Dict]:
    """
    Maps to: policy_retriever agent
    Task:    retrieve_company_policies
    """
    query = state["customer_query"]
    category = state.get("issue_category", "General")

    results = policy_service.search(query, n_results=3) if policy_service else []
    if not results:
        system = "You are a Company Policy Specialist. Identify applicable policies."
        human = f"Query: {query}\nCategory: {category}\nList 2-3 applicable policies and rules."
        content = await _acall_llm(system, human)
        results = [{"content": content, "score": 0.6, "source": "llm_policy_fallback"}]

    return results


async def _web_search_coordinate(state: SupportState) -> List[Dict]:
    """
    Maps to: web_search_coordinator agent
    Task:    coordinate_web_search
    """
    query = state["customer_query"]
    kb_results = state.get("kb_results", [])

    # Determine if web search is needed (low KB confidence)
    avg_score = (
        sum(r.get("score", 0) for r in kb_results) / len(kb_results)
        if kb_results else 0.0
    )

    if avg_score > 0.7:
        return [{"content": "Internal knowledge sufficient — web search skipped",
                 "score": 1.0, "source": "skipped_high_confidence"}]

    system = """You are an External Information Research Coordinator.
Simulate searching for current, relevant external information.
Return: {"search_needed": true/false, "key_findings": ["finding1", "finding2"], "sources": ["source1"]}"""
    human = f"Query: {query}\nInternal KB confidence: {avg_score:.2f}"
    raw = await _acall_llm(system, human)

    try:
        data = json.loads(raw)
        if data.get("search_needed"):
            return [{"content": f, "score": 0.7, "source": "web_search"}
                    for f in data.get("key_findings", [])]
    except Exception:
        pass

    return [{"content": f"External research for: {query}", "score": 0.5, "source": "web_search"}]


async def parallel_retrieval_node(
    state: SupportState,
    kb_service=None,
    policy_service=None,
) -> Dict[str, Any]:
    """
    Runs KB retrieval, policy retrieval, and web search SIMULTANEOUSLY.
    This is the key performance improvement over CrewAI's sequential execution.
    """
    start = time.time()

    # Fire all three concurrently — asyncio.gather is the magic here
    kb_task = asyncio.create_task(_retrieve_kb(state, kb_service))
    policy_task = asyncio.create_task(_retrieve_policies(state, policy_service))
    web_task = asyncio.create_task(_web_search_coordinate(state))

    kb_results, policy_results, web_results = await asyncio.gather(
        kb_task, policy_task, web_task
    )

    timings = dict(state.get("node_timings") or {})
    timings["parallel_retrieval"] = round(time.time() - start, 2)

    return {
        "kb_results": kb_results,
        "policy_results": policy_results,
        "web_results": web_results,
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 · INFORMATION FUSION
# ═══════════════════════════════════════════════════════════════════════════════
def fuse_information_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: information_fusion_agent
    Task:    information_fusion_task
    """
    start = time.time()
    kb = state.get("kb_results", [])
    policies = state.get("policy_results", [])
    web = state.get("web_results", [])
    session = state.get("session_context", {})

    kb_text = "\n".join(r.get("content", "")[:400] for r in kb[:3])
    policy_text = "\n".join(
        f"[{r.get('title', 'Policy')}]: {r.get('rules', r.get('content',''))[:300]}"
        for r in policies[:3]
    )
    web_text = "\n".join(r.get("content", "")[:300] for r in web[:2])

    system = """You are an Information Integration Specialist. Your job is to merge all sources
into a structured, policy-bound knowledge package for the response agent.

CRITICAL: Policy constraints are HARD LIMITS — the solution agent CANNOT go beyond them.

Return JSON:
{
  "primary_information": "most relevant synthesized content from KB",
  "hard_policy_constraints": "explicit list of what the agent CANNOT do (refund limits, auth limits, SLA limits)",
  "what_agent_can_offer": "what the agent IS authorised to offer within policy",
  "policy_constraints": "full policy rules that apply",
  "additional_context": "supporting details from web/external",
  "knowledge_gaps": ["gap1", "gap2"],
  "information_quality": 0.85,
  "solution_approach": "recommended approach WITHIN policy",
  "escalation_required": false,
  "escalation_reason": "why escalation is needed if true"
}
Return ONLY valid JSON."""

    human = f"""Customer Query: {state['customer_query']}
Classification: urgency={state.get('urgency_level')}, category={state.get('issue_category')}

KB Results (what we know):
{kb_text or 'None found'}

APPLICABLE COMPANY POLICIES (these are HARD LIMITS — cannot be exceeded):
{policy_text or 'Standard policies apply'}

External Information:
{web_text or 'No external data'}

Customer Profile: {json.dumps(session)}"""


    raw = _call_llm(system, human)
    try:
        fused = json.loads(raw)
    except Exception:
        fused = {
            "primary_information": kb_text or "General support guidance",
            "policy_constraints": policy_text or "Standard policies apply",
            "additional_context": web_text or "",
            "knowledge_gaps": [],
            "information_quality": 0.6,
            "solution_approach": "Standard resolution workflow",
        }

    timings = dict(state.get("node_timings") or {})
    timings["fuse_information"] = round(time.time() - start, 2)

    return {
        "fused_information": fused,
        "information_quality": float(fused.get("information_quality", 0.7)),
        "knowledge_gaps": fused.get("knowledge_gaps", []),
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 · SOLUTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_solution_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: solution_generator agent
    Task:    solution_generation_task
    POLICY ENFORCEMENT: solution MUST stay within retrieved policy limits.
    """
    start = time.time()
    fused = state.get("fused_information", {})
    policy_results = state.get("policy_results", [])
    attempts = state.get("qa_attempts", 0)

    # Build a strict policy constraint block from retrieved policies
    policy_constraints = []
    for p in policy_results[:5]:
        rules = p.get("rules") or p.get("content", "")
        title = p.get("title", "Policy")
        auth  = p.get("authorization_level", "agent")
        if rules:
            policy_constraints.append(f"[{title} | auth={auth}]: {rules}")

    policy_block = "\n".join(policy_constraints) if policy_constraints else "Standard company policies apply."

    retry_note = ""
    if attempts > 0:
        qa_result = state.get("qa_result", {})
        retry_note = f"\n⚠️ RETRY #{attempts}: Previous QA score was {state.get('qa_score', 0)}/10. Issues: {qa_result.get('improvement_notes', 'improve completeness and accuracy.')}"

    system = f"""You are a Solution Development Specialist. Generate a solution that STRICTLY follows company policies.

CRITICAL POLICY CONSTRAINTS — YOU CANNOT EXCEED THESE:
{policy_block}

MANDATORY RULES:
- NEVER promise a refund beyond what the policy allows (e.g. if policy says 30 days, do NOT offer refunds after 30 days)
- NEVER approve amounts beyond the agent authorization level
- NEVER make commitments that require supervisor/manager approval without flagging the need for escalation
- If the customer's request exceeds policy limits, clearly but empathetically explain the policy boundary
- Always tell the customer what CAN be done within policy, not just what cannot
- If escalation is required by policy, include that as a step
- CRITICAL: If the customer reports a damaged, leaking, or defective product and requests a return, refund, or replacement, you MUST ask them to provide photographic proof (images) before proceeding with any return processing.

Return JSON:
{{
  "solution_overview": "high-level summary",
  "primary_steps": ["step 1", "step 2", "step 3"],
  "prerequisites": ["what customer needs before starting"],
  "expected_outcome": "what success looks like",
  "alternative_approaches": ["alternative 1"],
  "risk_warnings": ["warning 1"],
  "verification_steps": ["how to confirm it worked"],
  "escalation_trigger": "when to escalate if this fails",
  "estimated_time": "5-10 minutes",
  "policy_boundaries_applied": ["policy limit 1 applied", "policy limit 2 applied"],
  "within_agent_authority": true
}}
Return ONLY valid JSON.{retry_note}"""

    human = f"""Query: {state['customer_query']}
Urgency: {state.get('urgency_level')} | Category: {state.get('issue_category')} | Complexity: {state.get('complexity')}

Integrated Knowledge:
{json.dumps(fused, indent=2)[:1200]}

Customer Expertise: {state.get('personalization', {}).get('expertise', 'Intermediate')}"""

    raw = _call_llm(system, human)
    try:
        solution = json.loads(raw)
    except Exception:
        solution = {
            "solution_overview": f"Resolution for: {state['customer_query'][:80]}",
            "primary_steps": [
                "1. Verify the reported issue against our records",
                "2. Apply resolution within applicable policy limits",
                "3. Confirm resolution with customer",
            ],
            "prerequisites": ["Account verification required"],
            "expected_outcome": "Issue resolved within policy guidelines",
            "alternative_approaches": [],
            "risk_warnings": ["Resolution subject to applicable company policies"],
            "verification_steps": ["Confirm the issue is resolved"],
            "escalation_trigger": "If request exceeds agent authorization level",
            "estimated_time": "10-15 minutes",
            "policy_boundaries_applied": ["Standard policies applied"],
            "within_agent_authority": True,
        }

    timings = dict(state.get("node_timings") or {})
    timings["generate_solution"] = round(time.time() - start, 2)

    return {
        "solution_package": solution,
        "primary_solution": solution.get("solution_overview", ""),
        "alternative_solutions": solution.get("alternative_approaches", []),
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 · DYNAMIC RESPONDER
# ═══════════════════════════════════════════════════════════════════════════════
def personalize_response_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: dynamic_responder agent
    Task:    dynamic_response_task
    """
    start = time.time()
    solution = state.get("solution_package", {})
    personalization = state.get("personalization", {})
    sentiment = state.get("sentiment", "Neutral")
    session = state.get("session_context", {})

    style = personalization.get("style", "Professional")
    expertise = personalization.get("expertise", "Intermediate")
    fmt = personalization.get("format", "DetailedSteps")

    empathy_map = {
        "Angry": "I completely understand your frustration, and I sincerely apologize for the inconvenience. ",
        "Frustrated": "I understand this has been a challenging experience, and I appreciate your patience. ",
        "Neutral": "Thank you for reaching out to us. ",
        "Satisfied": "Thank you for contacting us! ",
    }
    empathy_opener = empathy_map.get(sentiment, "Thank you for reaching out. ")

    system = f"""You are a Customer Communication Specialist. Transform the technical solution 
into a warm, personalized response.

Guidelines:
- Communication style: {style}
- Customer expertise level: {expertise}
- Response format: {fmt}
- Emotional tone: empathetic for {sentiment} customer
- Begin with: "{empathy_opener}"
- Use numbered steps if format is DetailedSteps
- Keep it conversational, not robotic
- End with an offer for further help

Return the final customer-facing response as plain text (no JSON)."""

    human = f"""Query: {state['customer_query']}

Solution Package:
{json.dumps(solution, indent=2)[:1200]}

Previous interaction context: {session.get('previous_issues_summary', 'None')}"""

    response = _call_llm(system, human)

    timings = dict(state.get("node_timings") or {})
    timings["personalize_response"] = round(time.time() - start, 2)

    return {
        "personalized_response": response,
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 · QA REVIEWER (with conditional retry loop)
# ═══════════════════════════════════════════════════════════════════════════════
def qa_review_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: qa_agent
    Task:    qa_review_task
    Score < 7 → retry generate_solution (max 2 times)
    KEY CHECK: Did the response stay within policy boundaries?
    """
    start = time.time()
    response   = state.get("personalized_response", "")
    query      = state["customer_query"]
    attempts   = state.get("qa_attempts", 0)
    solution   = state.get("solution_package", {})
    policy_results = state.get("policy_results", [])

    # Build policy summary for QA
    policy_summary = "\n".join(
        f"- [{p.get('title','Policy')}]: {p.get('rules', p.get('content',''))[:200]}"
        for p in policy_results[:4]
    ) or "Standard policies apply."

    system = f"""You are a Quality Assurance and Policy Compliance Specialist.
Your PRIMARY job: verify the response did NOT promise or offer anything beyond company policy.

APPLICABLE POLICIES:
{policy_summary}

REVIEW CRITERIA:
1. POLICY COMPLIANCE (most critical): Did the response stay strictly within policy limits?
   - No refund promised beyond allowed period?
   - No amount promised beyond agent authority?
   - No commitments requiring manager/supervisor without flagging escalation?
2. ACCURACY: Is information factually correct?
3. COMPLETENESS: Are all customer needs addressed?
4. TONE: Professional, empathetic, not robotic?
5. ACTIONABILITY: Are steps clear and executable?

Return JSON:
{{
  "overall_score": 8,
  "accuracy_score": 8,
  "completeness_score": 9,
  "tone_score": 8,
  "policy_compliance": true,
  "policy_violations": ["list any promises that exceed policy — empty if none"],
  "approval_status": "Approved|NeedsRevision|PolicyViolation|Escalate",
  "improvement_notes": "specific improvements if score < 7",
  "strengths": ["what works well"],
  "critical_issues": ["must fix items"]
}}
Return ONLY valid JSON."""

    human = f"""Original Query: {query}

Draft Response:
{response[:1500]}

Solution package used:
- Within agent authority: {solution.get('within_agent_authority', True)}
- Policy boundaries applied: {solution.get('policy_boundaries_applied', [])}"""

    raw = _call_llm(system, human)
    try:
        qa = json.loads(raw)
    except Exception:
        qa = {
            "overall_score": 7,
            "accuracy_score": 7,
            "completeness_score": 7,
            "tone_score": 7,
            "policy_compliance": True,
            "policy_violations": [],
            "approval_status": "Approved",
            "improvement_notes": "",
            "strengths": [],
            "critical_issues": [],
        }

    # Hard fail: policy violation forces retry regardless of score
    score = int(qa.get("overall_score", 7))
    if qa.get("policy_violations") and len(qa["policy_violations"]) > 0:
        score = min(score, 4)  # force retry if policy violated
        qa["improvement_notes"] = (
            f"POLICY VIOLATION detected: {qa['policy_violations']}. "
            "Revise to stay strictly within policy limits."
        )

    new_attempts = attempts + 1

    timings = dict(state.get("node_timings") or {})
    timings["qa_review"] = round(time.time() - start, 2)

    return {
        "qa_result": qa,
        "qa_score": score,
        "qa_attempts": new_attempts,
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 8 · CONVERSATION PERSISTER
# ═══════════════════════════════════════════════════════════════════════════════
def persist_conversation_node(
    state: SupportState,
    kb_service=None,
    cache_service=None,
) -> Dict[str, Any]:
    """
    Maps to: conversation_persister agent
    Task:    conversation_persistence_task
    Also stores to ChromaDB and caches the response.
    """
    start = time.time()
    query = state["customer_query"]
    response = state.get("personalized_response", "")
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "")
    category = state.get("issue_category", "General")
    urgency = state.get("urgency_level", "Medium")

    # Store in ChromaDB for future context
    if kb_service:
        try:
            kb_service.store_conversation(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response,
                metadata={
                    "category": category,
                    "urgency": urgency,
                    "qa_score": state.get("qa_score", 0),
                },
            )
        except Exception as e:
            logger.warning(f"Conversation persist to KB failed: {e}")

    # Cache the successful response in Redis
    if cache_service:
        try:
            cache_key = state.get("cache_key", "")
            if cache_key:
                cache_service.cache_response(
                    cache_key,
                    {
                        "final_response": response,
                        "category": category,
                        "urgency": urgency,
                        "qa_score": state.get("qa_score", 0),
                    },
                )
        except Exception as e:
            logger.warning(f"Response cache write failed: {e}")

    # Determine if escalation is needed
    special_flags = state.get("special_flags", [])
    qa_score = state.get("qa_score", 7)
    urgency = state.get("urgency_level", "Medium")

    needs_escalation = (
        "Escalation" in special_flags
        or "Legal" in special_flags
        or urgency == "Critical"
        or qa_score < 5
    )

    timings = dict(state.get("node_timings") or {})
    timings["persist_conversation"] = round(time.time() - start, 2)

    return {
        "conversation_persisted": True,
        "escalation_needed": needs_escalation,
        "final_response": response,
        "node_timings": timings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 9 · ESCALATION COORDINATOR (runs in parallel with CX Optimizer)
# ═══════════════════════════════════════════════════════════════════════════════
def escalation_coordinator_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: escalation_coordinator agent
    Only runs when escalation_needed = True
    """
    start = time.time()
    needs_escalation = state.get("escalation_needed", False)

    if not needs_escalation:
        timings = dict(state.get("node_timings") or {})
        timings["escalation"] = 0.0
        return {"escalation_report": {"status": "not_required"}, "node_timings": timings}

    query = state["customer_query"]
    urgency = state.get("urgency_level", "High")
    flags = state.get("special_flags", [])

    system = """You are an Escalation Management Expert. Create an escalation report.
Return JSON:
{
  "escalation_level": "Tier2|Specialist|Supervisor|Legal|Management",
  "reason": "why escalation is needed",
  "priority": "Immediate|High|Medium",
  "handoff_notes": "what the next team needs to know",
  "customer_communication": "what to tell the customer about escalation",
  "estimated_resolution_time": "e.g. 2-4 hours"
}
Return ONLY valid JSON."""

    human = f"Query: {query}\nUrgency: {urgency}\nFlags: {flags}\nQA Score: {state.get('qa_score', 0)}"
    raw = _call_llm(system, human)

    try:
        report = json.loads(raw)
    except Exception:
        report = {
            "escalation_level": "Tier2",
            "reason": f"Issue requires specialized attention: {urgency} urgency",
            "priority": "High",
            "handoff_notes": f"Customer query: {query[:200]}",
            "customer_communication": "A specialist will contact you within 2-4 hours.",
            "estimated_resolution_time": "2-4 hours",
        }

    timings = dict(state.get("node_timings") or {})
    timings["escalation"] = round(time.time() - start, 2)

    return {"escalation_report": report, "node_timings": timings}


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 10 · CX OPTIMIZER (runs in parallel with Escalation)
# ═══════════════════════════════════════════════════════════════════════════════
def cx_optimizer_node(state: SupportState) -> Dict[str, Any]:
    """
    Maps to: customer_experience_optimizer agent
    Analyzes the interaction for CX insights.
    """
    start = time.time()

    system = """You are a Customer Experience Analysis Specialist. Analyze this support interaction.
Return JSON:
{
  "satisfaction_prediction": 8.5,
  "cx_score": 8,
  "improvement_opportunities": ["opportunity1", "opportunity2"],
  "positive_aspects": ["what went well"],
  "proactive_suggestions": ["suggest1"],
  "knowledge_base_update_needed": false,
  "follow_up_recommended": false,
  "follow_up_timeline": "none"
}
Return ONLY valid JSON."""

    human = f"""Query: {state['customer_query']}
Category: {state.get('issue_category')} | Urgency: {state.get('urgency_level')}
Sentiment: {state.get('sentiment')} | QA Score: {state.get('qa_score', 0)}
Escalation Needed: {state.get('escalation_needed', False)}
Processing Time: {sum((state.get('node_timings') or {}).values()):.1f}s"""

    raw = _call_llm(system, human)
    try:
        cx_data = json.loads(raw)
    except Exception:
        cx_data = {
            "satisfaction_prediction": 7.5,
            "cx_score": 7,
            "improvement_opportunities": [],
            "positive_aspects": ["Issue addressed"],
            "follow_up_recommended": state.get("escalation_needed", False),
        }

    timings = dict(state.get("node_timings") or {})
    timings["cx_optimizer"] = round(time.time() - start, 2)

    # Compute total processing time
    total_time = sum(timings.values())

    return {
        "cx_optimization": cx_data,
        "processing_time": round(total_time, 2),
        "node_timings": timings,
    }
