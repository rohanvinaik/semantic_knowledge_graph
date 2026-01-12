"""
GSE Semantic Knowledge Graph

A Python library for extracting knowledge graphs from text using
Grounded Semantic Encoding (GSE).

Two extraction strategies available via cassettes:
- "semantic" (default): Knowledge-grounded using Wikipedia entity linking
- "grammar": Syntax-based using sentence structure and co-occurrence

Basic usage:
    from gse_graph import encode_text, extract_graph, resolve_entities

    text = "Einstein discovered relativity in 1905."
    encodings = encode_text(text)
    graph = extract_graph(encodings, text)  # Uses semantic grounding
    graph = resolve_entities(graph)

Using specific cassette:
    graph = extract_graph(encodings, text, cassette="grammar")

Direct cassette access:
    from gse_graph.cassettes import get_cassette
    cassette = get_cassette("semantic")
    graph = cassette.extract(encodings, text)
"""

from .structures import TokenEncoding, GraphNode, GraphEdge, SemanticGraph
from .encode import encode_text
from .extract import extract_graph
from .resolve import resolve_entities

# Cassette system
from .cassettes import get_cassette, ExtractionCassette, GrammarCooccurrenceCassette

# Try to import semantic cassette (optional dependency)
try:
    from .cassettes import SemanticGroundingCassette
    _HAS_SEMANTIC = True
except ImportError:
    SemanticGroundingCassette = None
    _HAS_SEMANTIC = False


__version__ = "0.2.0"

__all__ = [
    # Core structures
    "TokenEncoding",
    "GraphNode",
    "GraphEdge",
    "SemanticGraph",
    # Pipeline functions
    "encode_text",
    "extract_graph",
    "resolve_entities",
    # Cassette system
    "get_cassette",
    "ExtractionCassette",
    "GrammarCooccurrenceCassette",
    "SemanticGroundingCassette",
]
