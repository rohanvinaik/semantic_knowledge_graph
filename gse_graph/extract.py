"""
Extract graph structure from encoded tokens.

This module provides the main entry point for graph extraction.
It uses the cassette system to allow swapping between different
extraction strategies:

- "semantic" (default): Knowledge-grounded using Wikipedia entity linking
- "grammar": Syntax-based using sentence structure and co-occurrence

Usage:
    from gse_graph import extract_graph

    # Default: semantic grounding
    graph = extract_graph(encodings, text)

    # Explicit cassette selection
    graph = extract_graph(encodings, text, cassette="grammar")
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .structures import TokenEncoding, GraphNode, GraphEdge, SemanticGraph

# Import cassette system
from .cassettes import get_cassette, ExtractionCassette

# Module-level cassette cache
_default_cassette: Optional[ExtractionCassette] = None


def extract_graph(
    encodings: List[TokenEncoding],
    source_text: str,
    cassette: str = "semantic",
    **kwargs
) -> SemanticGraph:
    """
    Extract semantic graph from token encodings.

    Args:
        encodings: List of encoded tokens from encode_text()
        source_text: Original text (for metadata)
        cassette: Extraction strategy - "semantic" (default) or "grammar"
            - "semantic": Knowledge-grounded using Wikipedia entity linking,
              spreading activation, and anchor-based connectivity. Best for
              understanding actual concept meanings.
            - "grammar": Syntax-based using sentence structure and word
              co-occurrence. Fast, good for analyzing text structure.
        **kwargs: Passed to cassette constructor

    Returns:
        SemanticGraph with nodes and edges

    Example:
        >>> encodings = encode_text("Einstein discovered relativity.")
        >>> graph = extract_graph(encodings, text)  # Uses semantic grounding
        >>> graph = extract_graph(encodings, text, cassette="grammar")  # Uses syntax
    """
    global _default_cassette

    # Try semantic cassette first, fall back to grammar if unavailable
    try:
        extractor = get_cassette(cassette, **kwargs)
    except ImportError:
        if cassette == "semantic":
            # Fall back to grammar cassette
            extractor = get_cassette("grammar", **kwargs)
        else:
            raise

    return extractor.extract(encodings, source_text)


# Legacy functions moved to cassettes/grammar_cooccurrence.py
# Keeping minimal stubs for backwards compatibility
def _deprecated_warning(func_name: str):
    import warnings
    warnings.warn(
        f"{func_name} is deprecated. Use extract_graph() with cassette='grammar' instead.",
        DeprecationWarning,
        stacklevel=3
    )


def group_by_sentence(encodings: List[TokenEncoding]) -> Dict[int, List[TokenEncoding]]:
    """Deprecated: Use cassette system instead."""
    _deprecated_warning("group_by_sentence")
    by_sent = {}
    for tok in encodings:
        if tok.sentence_idx not in by_sent:
            by_sent[tok.sentence_idx] = []
        by_sent[tok.sentence_idx].append(tok)
    return by_sent
