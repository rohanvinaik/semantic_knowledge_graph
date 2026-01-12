from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
import numpy as np

@dataclass
class TokenEncoding:
    """Single token with its semantic encoding."""
    text: str                           # Original text
    lemma: str                          # Normalized form
    vector: np.ndarray                  # 8192-dim sparse ternary
    
    # Semantic decomposition
    word_primitives: Set[str]           # e.g., {"FEEL", "GOOD"}
    sentence_primitives: Set[str]       # e.g., {"SYN_SUBJ"}
    roles: Set[str]                     # e.g., {"agent", "experiencer"}
    
    # Bank activations (key for typing)
    bank_activations: Dict[str, float]  # e.g., {"MENTAL": 0.7, "ACTION": 0.3}
    
    # Position in text
    sentence_idx: int
    token_idx: int
    
    def is_entity(self) -> bool:
        """Entities have high SUBSTANTIVE bank or SOMEONE/SOMETHING primitives."""
        # Threshold adjusted for sparse vectors (approx 0.006 density per primitive)
        return (self.bank_activations.get("SUBSTANTIVES", 0) > 0.003 or
                bool(self.word_primitives & {"SOMEONE", "SOMETHING", "PEOPLE"}))
    
    def is_relation(self) -> bool:
        """Relations have high ACTION bank or are operators."""
        return (self.bank_activations.get("ACTION", 0) > 0.003 or
                bool(self.sentence_primitives & {"SYN_VERB"}))

@dataclass
class GraphNode:
    """Node in the semantic graph."""
    id: str                             # Unique identifier
    label: str                          # Display label
    vector: np.ndarray                  # 8192-dim (average of all mentions)
    
    # Aggregated from all mentions
    mentions: List[TokenEncoding]       # All tokens referring to this entity
    bank_activations: Dict[str, float]  # Averaged across mentions
    
    # 3D projection (computed later)
    position_3d: Optional[np.ndarray] = None  # [x, y, z]
    color_rgb: Optional[tuple] = None         # (r, g, b)
    
    # Optional wiki grounding
    wiki_entity_id: Optional[str] = None
    wiki_anchors: List[tuple] = field(default_factory=list)

@dataclass  
class GraphEdge:
    """Typed edge in the semantic graph."""
    source_id: str
    target_id: str
    
    # The relation itself has semantic content
    relation_text: str                  # e.g., "loves", "because"
    relation_vector: np.ndarray         # 8192-dim encoding of relation
    relation_banks: Dict[str, float]    # Bank activations of relation
    
    # Derived type (from bank activations)
    edge_type: str                      # "CAUSAL", "TEMPORAL", "ACTION", "MENTAL", etc.
    
    # Strength/confidence
    weight: float                       # Cosine similarity or count
    
    # Directionality (from operators)
    is_directed: bool = True
    direction: Optional[str] = None     # "CAUSE_TO_EFFECT", "BEFORE_TO_AFTER", etc.
    
    # For visualization
    color_rgb: Optional[tuple] = None
    
    @classmethod
    def infer_type(cls, banks: Dict[str, float]) -> str:
        """Infer edge type from bank activations."""
        if banks.get("LOGICAL", 0) > 0.003:
            return "CAUSAL"  # BECAUSE, IF, etc.
        if banks.get("TEMPORAL", 0) > 0.003:
            return "TEMPORAL"
        if banks.get("MENTAL", 0) > 0.003:
            return "MENTAL"
        if banks.get("SPATIAL", 0) > 0.003:
            return "SPATIAL"
        if banks.get("ACTION", 0) > 0.003:
            return "ACTION"
        if banks.get("EVALUATORS", 0) > 0.003:
            return "EVALUATIVE"
        return "GENERIC"

@dataclass
class SemanticGraph:
    """Complete semantic graph extracted from text."""
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    
    # Metadata
    source_text: str
    num_sentences: int
    
    # Analysis results (computed)
    clusters: Optional[Dict[int, List[str]]] = None  # cluster_id → node_ids
    
    def to_json(self) -> dict:
        """Export for frontend visualization."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "x": float(n.position_3d[0]) if n.position_3d is not None else 0.0,
                    "y": float(n.position_3d[1]) if n.position_3d is not None else 0.0,
                    "z": float(n.position_3d[2]) if n.position_3d is not None else 0.0,
                    "color": f"rgb{n.color_rgb}" if n.color_rgb else "rgb(128,128,128)",
                    "banks": n.bank_activations,
                    "mentions": len(n.mentions),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "label": e.relation_text,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "color": f"rgb{e.color_rgb}" if e.color_rgb else "rgb(100,100,100)",
                    "directed": e.is_directed,
                }
                for e in self.edges
            ],
            "metadata": {
                "source_length": len(self.source_text),
                "num_sentences": self.num_sentences,
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
            }
        }
