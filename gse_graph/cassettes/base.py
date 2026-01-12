"""
Base interface for extraction cassettes.

Cassettes are swappable modules that define how to extract
a semantic graph from encoded tokens. Different cassettes
provide different tradeoffs:

- Speed vs. accuracy
- Grammar-based vs. knowledge-grounded
- Local context vs. global knowledge
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..structures import TokenEncoding, SemanticGraph


@dataclass
class CassetteInfo:
    """Metadata about a cassette's capabilities."""
    name: str
    description: str
    strengths: List[str]
    weaknesses: List[str]
    requires_external: bool = False
    external_dependency: Optional[str] = None


class ExtractionCassette(ABC):
    """
    Abstract base class for graph extraction cassettes.

    Each cassette implements a different approach to extracting
    semantic graphs from GSE-encoded text.
    """

    @property
    @abstractmethod
    def info(self) -> CassetteInfo:
        """Return metadata about this cassette's capabilities."""
        pass

    @abstractmethod
    def extract(
        self,
        encodings: List[TokenEncoding],
        source_text: str,
        **kwargs
    ) -> SemanticGraph:
        """
        Extract a semantic graph from encoded tokens.

        Args:
            encodings: List of GSE-encoded tokens
            source_text: Original text (for metadata)
            **kwargs: Cassette-specific options

        Returns:
            SemanticGraph with nodes and edges
        """
        pass

    def benchmark(
        self,
        encodings: List[TokenEncoding],
        source_text: str,
        iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark this cassette's performance.

        Returns:
            Dict with timing, node/edge counts, etc.
        """
        import time

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            graph = self.extract(encodings, source_text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "cassette": self.info.name,
            "avg_time_ms": sum(times) / len(times) * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "tokens": len(encodings),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "tokens_per_sec": len(encodings) / (sum(times) / len(times)),
        }
