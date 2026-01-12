"""
Cassette System for Graph Extraction.

Cassettes are swappable extraction modules that provide different
approaches to building semantic graphs from text:

- GrammarCooccurrenceCassette: Syntax-based extraction using sentence
  structure and word co-occurrence. Fast, captures grammatical relations.

- SemanticGroundingCassette: Knowledge-grounded extraction using
  sparse-wiki entity linking and spreading activation. Captures actual
  semantic meaning and concept relationships.

Usage:
    from gse_graph.cassettes import get_cassette

    # Default: semantic grounding (most powerful)
    cassette = get_cassette()

    # Or explicitly choose:
    cassette = get_cassette("grammar")  # Grammar/co-occurrence
    cassette = get_cassette("semantic") # Semantic grounding (default)
"""

from .base import ExtractionCassette
from .grammar_cooccurrence import GrammarCooccurrenceCassette

# Semantic grounding is optional (requires sparse-wiki-grounding)
try:
    from .semantic_grounding import SemanticGroundingCassette
    _HAS_SEMANTIC = True
except ImportError:
    _HAS_SEMANTIC = False
    SemanticGroundingCassette = None


def get_cassette(
    cassette_type: str = "semantic",
    **kwargs
) -> ExtractionCassette:
    """
    Factory function to get an extraction cassette.

    Args:
        cassette_type: "semantic" (default) or "grammar"
        **kwargs: Passed to cassette constructor

    Returns:
        Configured ExtractionCassette instance
    """
    if cassette_type == "grammar":
        return GrammarCooccurrenceCassette(**kwargs)

    elif cassette_type == "semantic":
        if not _HAS_SEMANTIC:
            raise ImportError(
                "SemanticGroundingCassette requires sparse-wiki-grounding. "
                "Install it or use cassette_type='grammar'"
            )
        return SemanticGroundingCassette(**kwargs)

    else:
        raise ValueError(f"Unknown cassette type: {cassette_type}")


__all__ = [
    "ExtractionCassette",
    "GrammarCooccurrenceCassette",
    "SemanticGroundingCassette",
    "get_cassette",
]
