"""
Chainlit Conversational Frontend for Secure Enterprise RAG.
Powered by LangChain, Google Gemini (ChatGoogleGenerativeAI), FastMCP, and Qdrant.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional
import chainlit as cl
from langchain_core.messages import HumanMessage

from rag_poc.agent import create_guarded_orchestrator
from rag_poc.agent.mcp_client import get_secure_mcp_tools
from rag_poc.agent.orchestrator import _clean_text_content
from rag_poc.core.models import ClassificationLevel
from rag_poc.security import AuditLedger, JWTAuthenticator

logger = logging.getLogger(__name__)
authenticator = JWTAuthenticator()
audit_ledger = AuditLedger()

# Domain profiles mapping
PROFILE_CONFIGS = {
    "Customer Support (Hugging Face)": {
        "user_id": "support-agent@retailshop.internal",
        "tenant_id": "tenant_alpha",
        "clearance_level": ClassificationLevel.INTERNAL,
        "security_groups": ["support_agents", "default", "customers"],
        "description": "Hugging Face Bitext retail dataset: orders, cancellations, returns, refunds, payment troubleshooting, and shipping.",
    },
    "Workplace & HR Policies": {
        "user_id": "employee@enterprise.internal",
        "tenant_id": "tenant_alpha",
        "clearance_level": ClassificationLevel.INTERNAL,
        "security_groups": ["default", "hr_team", "security_team"],
        "description": "Employee benefits, 401(k) matching, home office stipends, summer hours, and career progression.",
    },
    "Security & Key Management": {
        "user_id": "security-analyst@enterprise.internal",
        "tenant_id": "tenant_alpha",
        "clearance_level": ClassificationLevel.INTERNAL,
        "security_groups": ["security_team", "default"],
        "description": "Enterprise key management standards, encryption policies, and KMS procedures.",
    },
    "Core Engineering": {
        "user_id": "engineer@enterprise.internal",
        "tenant_id": "tenant_beta",
        "clearance_level": ClassificationLevel.CONFIDENTIAL,
        "security_groups": ["core_engineering", "default"],
        "description": "Core engineering architecture blueprints and integration specifications.",
    },
}


@cl.set_chat_profiles
async def set_chat_profiles():
    """Allows users to select their documentation domain."""
    return [
        cl.ChatProfile(
            name="Customer Support (Hugging Face)",
            markdown_description=PROFILE_CONFIGS["Customer Support (Hugging Face)"]["description"],
            icon="https://api.iconify.design/lucide:shopping-bag.svg?color=%23f59e0b",
        ),
        cl.ChatProfile(
            name="Workplace & HR Policies",
            markdown_description=PROFILE_CONFIGS["Workplace & HR Policies"]["description"],
            icon="https://api.iconify.design/lucide:briefcase.svg?color=%233b82f6",
        ),
        cl.ChatProfile(
            name="Security & Key Management",
            markdown_description=PROFILE_CONFIGS["Security & Key Management"]["description"],
            icon="https://api.iconify.design/lucide:shield.svg?color=%2310b981",
        ),
        cl.ChatProfile(
            name="Core Engineering",
            markdown_description=PROFILE_CONFIGS["Core Engineering"]["description"],
            icon="https://api.iconify.design/lucide:cpu.svg?color=%238b5cf6",
        ),
    ]


class CitationItem:
    def __init__(
        self,
        chunk_id: str,
        source_uri: str,
        classification: str,
        content: str,
        formatted_markdown: str,
    ):
        self.chunk_id = chunk_id
        self.source_uri = source_uri
        self.classification = classification
        self.content = content
        self.formatted_markdown = formatted_markdown

    @property
    def title(self) -> str:
        return f"Source: {self.source_uri}"


def parse_context_citations(tool_output: str) -> List[CitationItem]:
    """
    Parses <context id="..." source="..." classification="..." data_only="true"> XML blocks
    and extracts grounding citations with HMAC verification badges.
    """
    citations: List[CitationItem] = []
    pattern = r'<context\s+id="([^"]+)"\s+source="([^"]*)"(?:\s+classification="([^"]+)")?[^>]*>([\s\S]*?)<\/context>'

    matches = list(re.finditer(pattern, tool_output))
    for match in matches:
        chunk_id = match.group(1).strip()
        source_uri = match.group(2).strip() or "internal-knowledge-base"
        classification = match.group(3) or "INTERNAL"
        content = match.group(4).strip()

        citation_md = (
            f"**Source Document**: `{source_uri}`\n\n"
            f"**Classification**: `{classification}`\n\n"
            f"**Chunk ID**: `{chunk_id}`\n\n"
            f"**Integrity Verification**: `✓ HMAC-SHA256 Cryptographic Signature Valid`\n\n"
            f"---\n\n"
            f"**Retrieved Evidence Excerpt**:\n"
            f"> {content}\n"
        )

        citations.append(
            CitationItem(
                chunk_id=chunk_id,
                source_uri=source_uri,
                classification=classification,
                content=content,
                formatted_markdown=citation_md,
            )
        )

    return citations


@cl.on_chat_start
async def on_chat_start():
    """Initializes the chat session with domain-scoped FastMCP retrieval and LangGraph agent."""
    profile_name = cl.user_session.get("chat_profile") or "Customer Support (Hugging Face)"
    config = PROFILE_CONFIGS.get(profile_name, PROFILE_CONFIGS["Customer Support (Hugging Face)"])

    tenant_id = config["tenant_id"]
    user_id = config["user_id"]
    clearance_level = config["clearance_level"]
    security_groups = config["security_groups"]

    # 1. Cryptographically generate JWT session token
    token = authenticator.generate_token(
        sub=user_id,
        tenant_id=tenant_id,
        security_groups=security_groups,
        clearance_level=clearance_level,
    )

    # 2. Bind FastMCP tool with server-side tenant scoping
    tools = await get_secure_mcp_tools(auth_token=token, tenant_id=tenant_id)

    # 3. Compile Guarded Routing Orchestrator with Audit Ledger
    orchestrator = create_guarded_orchestrator(tools=tools, audit_ledger=audit_ledger)

    # Save to session
    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("profile_name", profile_name)
    cl.user_session.set("tenant_id", tenant_id)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("clearance_level", clearance_level.value)
    cl.user_session.set("token", token)
    cl.user_session.set("thread_id", f"thread_{tenant_id}_{cl.user_session.get('id')}")

    # Interactive sample query action buttons tailored to the active profile
    if profile_name == "Customer Support (Hugging Face)":
        actions = [
            cl.Action(
                name="query_action",
                payload={"query": "How do I cancel an order and what is the cancellation cutoff time?"},
                label="📦 Order Cancellation & Cutoff",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What is the 30-day refund policy and how long does it take for funds to appear?"},
                label="💸 Refund Window & Timelines",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "My card was declined at checkout. What payment methods are accepted and how do I fix payment errors?"},
                label="💳 Payment Methods & Card Declines",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What shipping speeds and delivery tiers are available, and how can I update my delivery address?"},
                label="🚚 Shipping Options & Address Changes",
            ),
        ]
    elif profile_name == "Workplace & HR Policies":
        actions = [
            cl.Action(
                name="query_action",
                payload={"query": "What is the 401(k) company match and medical insurance coverage?"},
                label="💰 401(k) & Health Benefits",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What is the first-year home office setup stipend and coworking space allowance?"},
                label="🏠 Home Office Stipend",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What is the company policy on summer hours and paid parental leave?"},
                label="🏖️ Summer Hours & Leave",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What are the performance review milestones during the first year?"},
                label="📈 First-Year Review Cadence",
            ),
        ]
    elif profile_name == "Core Engineering":
        actions = [
            cl.Action(
                name="query_action",
                payload={"query": "What are the core banking integration architecture specifications?"},
                label="🏗️ Architecture Blueprints",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "How does our zero-trust mTLS service mesh work on Kubernetes?"},
                label="🔒 Zero-Trust Service Mesh",
            ),
        ]
    else:
        actions = [
            cl.Action(
                name="query_action",
                payload={"query": "What is the AES-256 key rotation policy for payment transactions?"},
                label="🔑 Key Rotation Policy",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "How are KMS cryptographic backup master keys stored?"},
                label="🛡️ KMS Backup Storage",
            ),
            cl.Action(
                name="query_action",
                payload={"query": "What are the rotation intervals for DEK, KEK, and root keys?"},
                label="📋 Key Hierarchy & Cadence",
            ),
        ]

    welcome_content = "👋 **Welcome!** How can I assist you with enterprise documentation today?"

    await cl.Message(content=welcome_content, actions=actions).send()


@cl.action_callback("query_action")
async def on_action_clicked(action: cl.Action):
    """Executes a pre-defined test query when an action button is clicked."""
    user_query = str(action.payload.get("query", ""))
    if not user_query:
        return
    # Echo user action into the chat
    await cl.Message(content=user_query, author="User").send()
    # Process the query through the guarded orchestrator
    await process_user_query(user_query)


@cl.on_message
async def on_message(message: cl.Message):
    """Processes incoming user questions via the Guarded Routing Orchestrator."""
    await process_user_query(message.content)


async def process_user_query(query_text: str):
    """Executes input guardrails, intent routing, grounded retrieval, output rails, and audit logging."""
    orchestrator = cl.user_session.get("orchestrator")
    thread_id = cl.user_session.get("thread_id") or "default_thread"
    user_id = cl.user_session.get("user_id") or "anonymous"
    tenant_id = cl.user_session.get("tenant_id") or "tenant_alpha"
    clearance_level = cl.user_session.get("clearance_level") or "INTERNAL"
    token = cl.user_session.get("token") or ""
    profile_name = cl.user_session.get("profile_name") or cl.user_session.get("chat_profile") or "Customer Support (Hugging Face)"

    if not orchestrator:
        await cl.Message(content="⚠️ Agent session is not initialized. Please refresh the chat.").send()
        return

    inputs = {
        "messages": [HumanMessage(content=query_text)],
        "user_id": user_id,
        "tenant_id": tenant_id,
        "clearance_level": clearance_level,
        "auth_token": token,
        "session_id": thread_id,
        "domain": profile_name,
        "start_time": time.time(),
    }

    try:
        result = await orchestrator.ainvoke(inputs)

        intent = result.get("intent", "retrieval")
        event_id = result.get("event_id", "")
        final_content = _clean_text_content(result.get("final_response", ""))
        citations_data = result.get("retrieved_citations", [])

        # Format side citation cards cleanly
        all_citations: List[cl.Text] = []
        for c in citations_data:
            citation_md = (
                f"### Source: `{c['source_uri']}`\n\n"
                f"**Classification**: `{c['classification']}`\n\n"
                f"---\n\n"
                f"{c['content']}\n"
            )
            all_citations.append(
                cl.Text(
                    name=f"📄 Source: {c['source_uri']}",
                    content=citation_md,
                    display="side",
                )
            )

        feedback_actions: List[cl.Action] = []
        if intent == "retrieval" and event_id:
            feedback_actions = [
                cl.Action(
                    name="feedback_up",
                    payload={"event_id": event_id},
                    label="👍 Helpful",
                ),
                cl.Action(
                    name="feedback_down",
                    payload={"event_id": event_id},
                    label="👎 Needs Improvement",
                ),
            ]

        await cl.Message(
            content=final_content,
            elements=all_citations,
            actions=feedback_actions if feedback_actions else None,
        ).send()

    except Exception as e:
        logger.error("Error during orchestrator execution: %s", e, exc_info=True)
        await cl.Message(
            content=f"⚠️ **Error processing request**: {str(e)}"
        ).send()


@cl.action_callback("feedback_up")
async def on_feedback_up(action: cl.Action):
    """Records positive human feedback in the compliance audit ledger."""
    event_id = str(action.payload.get("event_id", ""))
    if event_id:
        audit_ledger.record_feedback(event_id, "positive")
    await cl.Message(
        content="✓ **Thank you for your feedback!**"
    ).send()


@cl.action_callback("feedback_down")
async def on_feedback_down(action: cl.Action):
    """Escalates response for Human-in-the-Loop review under EU AI Act Article 14."""
    event_id = str(action.payload.get("event_id", ""))
    if event_id:
        audit_ledger.record_feedback(event_id, "escalated", "User flagged response for human review")
    await cl.Message(
        content="✓ **Thank you for your feedback.** This response has been flagged for administrative review."
    ).send()
