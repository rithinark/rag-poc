"""
Guarded Routing Orchestrator for Enterprise RAG.
Orchestrates Input Guardrails (Meta Prompt Guard ONNX + Presidio PII),
Intent Routing (Conversational vs. Document Retrieval), ReAct Retrieval Agent (FastMCP),
Output Guardrails (EchoLeak & Secret Scrubbing), and Compliance Audit Logging.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag_poc.core.config import settings
from rag_poc.core.models import RetrievedPassage
from rag_poc.guardrails.input_rails import AsyncInputGuardrailPipeline
from rag_poc.guardrails.output_rails import OutputRailChecker
from rag_poc.agent.chatbot import get_llm, create_enterprise_chatbot
from rag_poc.agent.intent_router import BaseIntentClassifier, LLMIntentClassifier
from rag_poc.security.audit import AuditLedger, AuditRecord

logger = logging.getLogger(__name__)


def _clean_text_content(content: Any) -> str:
    """Extracts clean plain text from strings, lists of dictionaries, or content objects."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item and item["text"]:
                parts.append(str(item["text"]))
            elif hasattr(item, "text") and item.text:
                parts.append(str(item.text))
        if parts:
            return "".join(parts)
    return str(content)


DOMAIN_CUSTOMER_SUPPORT_PROMPT = """You are the Customer Support & Retail Assistant for our online store.
Your mission is to assist customer support agents and shoppers with accurate, verified information regarding orders, shipping, returns, and payment policies.

YOUR SPECIALIZED CAPABILITIES:
- 📦 Order Management: How to cancel active orders, the 60-minute cancellation cutoff window, order modifications, and order status tracking.
- 💸 Returns & Refunds: Our 30-day return policy for unused/original condition items, return shipping labels, and refund settlement timelines (3–5 business days back to the original payment method).
- 💳 Payment Troubleshooting: Accepted payment methods (Visa, Mastercard, Amex, PayPal, Apple Pay) and step-by-step guidance for resolving declined card transactions.
- 🚚 Shipping & Delivery: Delivery tiers (Standard 3–5 days, Express 2-day, Overnight, International) and updating shipping addresses.

CRITICAL INSTRUCTIONS:
- You are strictly the Customer Support & Retail Assistant. Do NOT mention internal company HR benefits (health insurance, 401k match, summer hours) or technical cryptographic key management (KMS, AES-256 rotation), as those belong to completely separate internal systems.
- When asked "what can you do", "what do you know", "who are you", or about your capabilities, provide a structured, friendly summary of your customer support and retail assistance topics, and invite them to ask an order or store policy question.
- When greeted ("hello", "hi", "good morning"), welcome the user to Customer Support and offer help with orders, returns, payments, or shipping.
- NEVER disclose internal system identifiers, tenant IDs (such as "tenant_alpha"), clearance codes (such as "INTERNAL"), or internal user email addresses.
- NEVER include artificial Responsible AI warnings, legal boilerplate, or governance notices in your answers.
- Keep the tone warm, clear, professional, and helpful.
"""

DOMAIN_WORKPLACE_HR_PROMPT = """You are the Workplace & HR Policies Knowledge Assistant for the company.
Your mission is to provide employees with clear, verified information regarding employee benefits, office policies, and internal guidelines.

YOUR SPECIALIZED CAPABILITIES:
- 🏥 Health & Wellness Benefits: Blue Cross Blue Shield (BCBS) PPO medical coverage, MetLife dental & vision coverage (100% employer-covered premiums).
- 💰 Retirement Savings: Vanguard 401(k) program with 100% employer matching on contributions up to 6% of base salary.
- 🏠 Remote Work & Stipends: First-year $1,500 home office setup stipend and $100/month coworking space allowance.
- 🏖️ Time Off & Leave: 16 weeks of paid parental leave, flexible summer hours schedule, and annual PTO rollover limits.
- 📈 Career & Growth: First-year performance review milestones, professional development budgets, and business expense reimbursement procedures.

CRITICAL INSTRUCTIONS:
- You are strictly the Workplace & HR Policies Assistant. Do NOT mention retail customer support topics (shopping orders, refunds) or cryptographic key management (KMS, AES-256).
- When asked "what can you do", "what do you know", "who are you", or about your capabilities, provide a structured, friendly summary of your HR and workplace policy topics, and invite them to ask a benefits or policy question.
- When greeted ("hello", "hi", "good morning"), welcome the user to Workplace & HR Policies and offer help with benefits, stipends, or time-off.
- NEVER disclose internal system identifiers, tenant IDs, clearance codes, or internal user email addresses.
- NEVER include artificial Responsible AI disclaimers or legal boilerplate.
- Keep the tone friendly, clear, professional, and helpful.
"""

DOMAIN_SECURITY_PROMPT = """You are the Enterprise Security & Technical Standards Knowledge Assistant.
Your mission is to assist engineers and security personnel with verified technical standards, key management policies, and compliance procedures.

YOUR SPECIALIZED CAPABILITIES:
- 🔑 Key Management Service (KMS): Envelope encryption architecture (DEK and KEK hierarchy), master key storage, and Hardware Security Module (HSM) deployment.
- 🔄 Cryptographic Key Rotation: AES-256 90-day automatic key rotation schedules, 30-day DEK in-memory lifespan limits, and emergency key destruction SOPs.
- 🛡️ Governance & Compliance: Multi-party split-knowledge quorum recovery procedures, and alignment with EU AI Act and NIST AI RMF standards.

CRITICAL INSTRUCTIONS:
- You are strictly the Enterprise Security & Technical Standards Assistant. Do NOT mention retail shopping or HR benefits.
- When asked "what can you do", "what do you know", "who are you", or about your capabilities, provide a concise, structured overview of your technical and cryptographic standards coverage.
- When greeted ("hello", "hi", "good morning"), welcome the user and offer assistance with security standards, encryption, or key management.
- NEVER disclose internal system identifiers, tenant IDs, clearance codes, or internal user email addresses.
- NEVER leak raw secret keys or tokens.
- NEVER include artificial Responsible AI disclaimers or legal boilerplate.
- Keep the tone precise, authoritative, professional, and helpful.
"""

DOMAIN_ENGINEERING_PROMPT = """You are the Core Engineering Architecture Assistant.
Your mission is to assist engineering teams with technical architecture blueprints, service mesh configurations, and integration specifications.

YOUR SPECIALIZED CAPABILITIES:
- 🏗️ Core Banking Architecture: System architecture blueprints and core integration specifications.
- 🔒 Service Mesh & Zero Trust: Mutual TLS (mTLS) zero-trust service mesh configuration and Kubernetes deployments.

CRITICAL INSTRUCTIONS:
- Focus strictly on Core Engineering Architecture.
- NEVER disclose internal system identifiers or raw credentials.
"""

GENERAL_ENTERPRISE_PROMPT = """You are an authoritative and helpful Enterprise Knowledge Assistant.
Your mission is to provide accurate, verified, and friendly information across our enterprise knowledge base.

CORE KNOWLEDGE DOMAINS:
1. 🛍️ Customer Support & Retail: Order cancellations, 30-day refund window, payment methods, delivery tiers, and shipping address updates.
2. 🏢 Workplace & HR Policies: Medical (BCBS PPO), dental & vision (MetLife), Vanguard 401(k) 6% match, $1,500 home office stipend, summer hours, and leave policies.
3. 🛡️ Enterprise Security & Technical Standards: KMS architecture, envelope encryption (DEK/KEK), AES-256 90-day rotation, and AI compliance standards.

CRITICAL INSTRUCTIONS:
- When asked "what do you know" or "what can you do" without a specific domain context, provide a structured overview of these 3 domains and invite the user to choose an area.
- NEVER disclose internal system identifiers (e.g. tenant_alpha), clearance classifications, or internal user emails.
- NEVER include artificial Responsible AI disclaimers or legal boilerplate.
- Keep the tone friendly, clear, professional, and well-structured.
"""


def get_conversational_system_prompt(domain: Optional[str] = None, query: str = "") -> str:
    """Selects the specialized conversational prompt based on domain or query context."""
    domain_str = (domain or "").lower()
    query_str = (query or "").lower()

    if "customer support" in domain_str or "retail" in domain_str or "customer support" in query_str or "order" in query_str:
        return DOMAIN_CUSTOMER_SUPPORT_PROMPT
    elif "workplace" in domain_str or "hr" in domain_str or "human resources" in query_str or "benefits" in query_str:
        return DOMAIN_WORKPLACE_HR_PROMPT
    elif "security" in domain_str or "kms" in domain_str or "key management" in query_str:
        return DOMAIN_SECURITY_PROMPT
    elif "engineering" in domain_str or "core engineering" in query_str:
        return DOMAIN_ENGINEERING_PROMPT
    return GENERAL_ENTERPRISE_PROMPT


CONVERSATIONAL_SYSTEM_PROMPT = GENERAL_ENTERPRISE_PROMPT



class CitationDetail(BaseModel):
    chunk_id: str
    source_uri: str
    classification: str
    content: str
    formatted_markdown: str


class OrchestratorState(TypedDict):
    """Execution state passed through the Guarded Routing Workflow."""
    messages: List[BaseMessage]
    user_id: str
    tenant_id: str
    clearance_level: str
    auth_token: Optional[str]
    session_id: str
    domain: Optional[str]

    # Input guardrail verdict
    is_safe: bool
    rejection_reason: Optional[str]
    sanitized_query: str
    detected_pii: List[str]

    # Routing intent
    intent: Literal["refusal", "conversational", "retrieval"]

    # Retrieval evidence & citations
    retrieved_citations: List[Dict[str, Any]]
    retrieved_chunks: List[str]

    # Output draft & final response
    draft_response: str
    final_response: str
    output_is_safe: bool
    output_rejection_reason: Optional[str]
    event_id: str
    start_time: float


class GuardedEnterpriseOrchestrator:
    """
    Coordinates multi-stage enterprise safety, routing, retrieval, and audit logging.
    """

    def __init__(
        self,
        tools: List[Any],
        react_agent: Optional[Any] = None,
        llm: Optional[BaseChatModel] = None,
        intent_classifier: Optional[BaseIntentClassifier] = None,
        audit_ledger: Optional[AuditLedger] = None,
        input_rails: Optional[AsyncInputGuardrailPipeline] = None,
        output_rails: Optional[OutputRailChecker] = None,
        checkpointer: Optional[Any] = None,
    ):
        self.tools = tools
        self.llm = llm
        self.intent_classifier = intent_classifier or LLMIntentClassifier(llm=self.llm)
        self.audit_ledger = audit_ledger or AuditLedger()
        self.input_rails = input_rails or AsyncInputGuardrailPipeline()
        self.output_rails = output_rails or OutputRailChecker()
        self.checkpointer = checkpointer or MemorySaver()
        self.react_agent = react_agent or create_enterprise_chatbot(
            tools=self.tools,
            checkpointer=self.checkpointer,
            llm=self.llm,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(OrchestratorState)

        # 1. Define nodes
        builder.add_node("input_guardrail", self._input_guardrail_node)
        builder.add_node("intent_classification", self._intent_classification_node)
        builder.add_node("refusal", self._refusal_node)
        builder.add_node("conversational", self._conversational_node)
        builder.add_node("retrieval_agent", self._retrieval_agent_node)
        builder.add_node("output_guardrail", self._output_guardrail_node)

        # 2. Define edges & routing
        builder.add_edge(START, "input_guardrail")
        builder.add_conditional_edges(
            "input_guardrail",
            self._route_after_input_guardrail,
            {
                "refusal": "refusal",
                "classify": "intent_classification",
            },
        )
        builder.add_conditional_edges(
            "intent_classification",
            self._route_after_intent_classification,
            {
                "conversational": "conversational",
                "retrieval": "retrieval_agent",
            },
        )
        builder.add_edge("refusal", "output_guardrail")
        builder.add_edge("conversational", "output_guardrail")
        builder.add_edge("retrieval_agent", "output_guardrail")
        builder.add_edge("output_guardrail", END)

        return builder.compile()

    # --------------------------------------------------------------------------
    # Graph Nodes
    # --------------------------------------------------------------------------

    def _input_guardrail_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Evaluates raw prompt via Meta Prompt Guard 86M and Presidio PII analyzer."""
        raw_prompt = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage) or (isinstance(msg, tuple) and msg[0] == "user"):
                raw_prompt = msg.content if isinstance(msg, HumanMessage) else msg[1]
                break

        if not raw_prompt:
            raw_prompt = "Hello"

        verdict = self.input_rails.process(raw_prompt)

        return {
            "is_safe": verdict.is_safe,
            "rejection_reason": verdict.rejection_reason,
            "sanitized_query": verdict.sanitized_query,
            "detected_pii": list(verdict.detected_pii_entities),
            "start_time": state.get("start_time") or time.time(),
        }

    def _route_after_input_guardrail(self, state: OrchestratorState) -> Literal["refusal", "classify"]:
        """Diverts unsafe prompts directly to refusal, or advances safe queries to intent classification."""
        if not state.get("is_safe", True):
            return "refusal"
        return "classify"

    async def _intent_classification_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Classifies the sanitized query via the model-based intent classifier."""
        query = state.get("sanitized_query", "")
        intent = await self.intent_classifier.aclassify(query)
        return {"intent": intent}

    def _route_after_intent_classification(self, state: OrchestratorState) -> Literal["conversational", "retrieval"]:
        """Routes classified query to conversational exchange or grounded vector retrieval."""
        intent = state.get("intent", "conversational")
        if intent == "retrieval":
            return "retrieval"
        return "conversational"

    def _refusal_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Constructs an authoritative Responsible AI refusal without calling the LLM."""
        reason = state.get("rejection_reason") or "Adversarial prompt injection pattern detected."
        content = (
            f"⚠️ **Request Declined (Security Policy Violation)**\n\n"
            f"Your request was flagged by **Meta Prompt Guard 86M**:\n"
            f"> *{reason}*\n\n"
            f"In accordance with enterprise Responsible AI governance and EU AI Act Article 15 standards, "
            f"adversarial system overrides and unauthorized scope queries are blocked."
        )
        return {
            "intent": "refusal",
            "draft_response": content,
            "messages": [AIMessage(content=content)],
        }

    async def _conversational_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Provides lightweight conversational responses without invoking vector retrieval."""
        query = state.get("sanitized_query", "")
        domain = state.get("domain")

        llm = self.llm or get_llm()
        system_prompt = get_conversational_system_prompt(domain=domain, query=query)
        system_msg = SystemMessage(content=system_prompt)
        response = await llm.ainvoke([system_msg, HumanMessage(content=query)])
        content = _clean_text_content(response.content)

        return {
            "intent": "conversational",
            "draft_response": content,
            "messages": [AIMessage(content=content)],
        }

    async def _retrieval_agent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Executes grounded ReAct agent equipped with FastMCP scoped retrieval."""
        query = state.get("sanitized_query", "")
        thread_id = state.get("session_id") or "default_thread"
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=query)]}

        result = await self.react_agent.ainvoke(inputs, config)
        messages = result.get("messages", [])

        # Extract final answer from agent messages
        draft = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                draft = _clean_text_content(m.content)
                break

        if not draft:
            draft = "I cannot satisfy this request because verified enterprise documentation does not contain sufficient evidence."

        # Extract citations from tool execution messages
        citations: List[Dict[str, Any]] = []
        chunk_ids: List[str] = []
        pattern = re.compile(
            r'<context id="([^"]+)" source="([^"]+)"(?: classification="([^"]*)")?[^>]*>(.*?)</context>',
            re.DOTALL | re.IGNORECASE,
        )

        for m in messages:
            content_str = str(m.content) if hasattr(m, "content") else ""
            for match in pattern.finditer(content_str):
                cid = match.group(1)
                s_uri = match.group(2)
                cls_lvl = match.group(3) or "INTERNAL"
                c_text = match.group(4).strip()
                chunk_ids.append(cid)
                citations.append({
                    "chunk_id": cid,
                    "source_uri": s_uri,
                    "classification": cls_lvl,
                    "content": c_text,
                })

        return {
            "intent": "retrieval",
            "draft_response": draft,
            "retrieved_citations": citations,
            "retrieved_chunks": chunk_ids,
            "messages": [AIMessage(content=draft)],
        }

    def _output_guardrail_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Runs OutputRailChecker and logs the completed interaction to AuditLedger."""
        draft = state.get("draft_response", "")
        intent = state.get("intent", "retrieval")
        citations_data = state.get("retrieved_citations", [])

        # 1. Output Rail: Grounding verification only for technical retrieval intent
        if intent == "retrieval":
            evidence: List[RetrievedPassage] = [
                RetrievedPassage(
                    chunk_id=c["chunk_id"],
                    document_id=c["source_uri"],
                    content=c["content"],
                    tenant_id=state.get("tenant_id", "tenant_alpha"),
                    similarity_score=1.0,
                )
                for c in citations_data
            ]
            verdict = self.output_rails.evaluate(draft, evidence)
            cleaned_text = verdict.sanitized_output
            output_is_safe = verdict.passed
            rejection_reason = verdict.refusal_reason if not verdict.passed else None
        else:
            # For refusal and conversational intents, sanitize exfiltration channels without document grounding
            cleaned_text, exfil_detected = self.output_rails.sanitize_exfiltration_channels(draft)
            output_is_safe = not exfil_detected
            rejection_reason = "Covert exfiltration channel detected in output." if exfil_detected else None

        # 2. Secret & API Key Sanitizer (prevent raw credentials from leaking)
        secret_pattern = re.compile(
            r"(AIza[0-9A-Za-z_-]{25,45}|super-secret[^\s]+|super-secure[^\s]+)",
            re.IGNORECASE,
        )
        if secret_pattern.search(cleaned_text):
            cleaned_text = secret_pattern.sub("[CONFIDENTIAL_CREDENTIAL_MASKED]", cleaned_text)
            output_is_safe = False
            rejection_reason = "Credential or secret token detected in output."

        final_text = cleaned_text

        # 3. Record to Audit Ledger
        start_time = state.get("start_time") or time.time()
        latency_ms = (time.time() - start_time) * 1000.0

        raw_prompt = ""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                raw_prompt = str(msg.content)
                break

        audit_rec = AuditRecord(
            session_id=state.get("session_id", "default_session"),
            user_id=state.get("user_id", "anonymous"),
            tenant_id=state.get("tenant_id", "tenant_alpha"),
            clearance_level=state.get("clearance_level", "INTERNAL"),
            raw_prompt=raw_prompt or state.get("sanitized_query", ""),
            sanitized_prompt=state.get("sanitized_query", ""),
            input_is_safe=state.get("is_safe", True),
            input_rejection_reason=state.get("rejection_reason"),
            intent=intent,
            chunks_retrieved=state.get("retrieved_chunks", []),
            output_is_safe=output_is_safe,
            output_rejection_reason=rejection_reason,
            disclaimer_appended=False,
            latency_ms=round(latency_ms, 2),
        )
        logged_record = self.audit_ledger.log_event(audit_rec)

        return {
            "final_response": final_text,
            "output_is_safe": output_is_safe,
            "output_rejection_reason": rejection_reason,
            "event_id": logged_record.event_id,
        }

    async def ainvoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes the guarded orchestrator asynchronously."""
        return await self.graph.ainvoke(state, config=config)


def create_guarded_orchestrator(
    tools: List[Any],
    react_agent: Optional[Any] = None,
    llm: Optional[BaseChatModel] = None,
    intent_classifier: Optional[BaseIntentClassifier] = None,
    audit_ledger: Optional[AuditLedger] = None,
    input_rails: Optional[AsyncInputGuardrailPipeline] = None,
    output_rails: Optional[OutputRailChecker] = None,
    checkpointer: Optional[Any] = None,
) -> GuardedEnterpriseOrchestrator:
    """Factory creating a production-grade Guarded Routing Orchestrator."""
    return GuardedEnterpriseOrchestrator(
        tools=tools,
        react_agent=react_agent,
        llm=llm,
        intent_classifier=intent_classifier,
        audit_ledger=audit_ledger,
        input_rails=input_rails,
        output_rails=output_rails,
        checkpointer=checkpointer,
    )
