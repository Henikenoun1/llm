"""
Backward-compatible wrapper around the new tools and RAG modules.
"""
from __future__ import annotations

from .rag import VectorRAGRetriever as KBRetriever
from .tools import get_tool_registry


def execute_tool(name: str, args: dict):
    return get_tool_registry().execute(name, args)


LENS_CATALOG = get_tool_registry().kb.lens_catalog
STORES = get_tool_registry().kb.stores
POLICIES = get_tool_registry().kb.policies

