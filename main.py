"""
Root entrypoint for Secure RAG POC.
Invokes the end-to-end demonstration.
"""

import sys
import os
import asyncio

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_poc.main import run_demonstration

if __name__ == "__main__":
    asyncio.run(run_demonstration())
