"""AutoStream Agentic Workflow — Core Agent Package.

This package implements a multi-node LangGraph pipeline for:
  - Intent classification (greeting / product query / high-intent lead)
  - RAG-powered knowledge retrieval from a local Markdown knowledge base
  - Stateful lead qualification with field-by-field collection
  - Mock lead capture tool execution upon full qualification
"""
