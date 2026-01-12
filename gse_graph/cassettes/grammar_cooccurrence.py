"""
Grammar & Co-occurrence Cassette.

Extracts graph structure based on:
1. Syntactic sentence structure (subject-verb-object)
2. Word co-occurrence within sentences
3. Bank activations for edge typing

This is a fast, syntax-aware extraction that captures grammatical
relationships but does NOT understand the actual meaning of concepts.

Use cases:
- Quick analysis of text structure
- Grammar/syntax visualization
- Baseline for comparison with semantic methods
"""

from typing import List, Dict, Optional
import numpy as np

from .base import ExtractionCassette, CassetteInfo
from ..structures import TokenEncoding, GraphNode, GraphEdge, SemanticGraph


class GrammarCooccurrenceCassette(ExtractionCassette):
    """
    Syntax-based graph extraction using sentence structure and co-occurrence.

    This cassette extracts:
    - Entities: Tokens with high SUBSTANTIVES bank or entity primitives
    - Relations: Based on syntactic roles (SYN_SUBJ, SYN_OBJ) or word order
    - Edge types: Inferred from bank activations (CAUSAL, TEMPORAL, etc.)

    Strengths:
    - Fast extraction (~19k edges/sec)
    - Captures grammatical structure
    - No external dependencies

    Weaknesses:
    - No semantic understanding of concepts
    - Proper nouns/domain terms have empty primitives
    - Edges are grammatical, not semantic relationships
    """

    @property
    def info(self) -> CassetteInfo:
        return CassetteInfo(
            name="grammar_cooccurrence",
            description="Syntax-based extraction using sentence structure and word co-occurrence",
            strengths=[
                "Fast extraction (~19k edges/sec)",
                "Captures grammatical relationships",
                "No external dependencies",
                "Good for syntax analysis",
            ],
            weaknesses=[
                "No semantic understanding of OOV concepts",
                "Proper nouns have empty primitives",
                "Edges represent grammar, not meaning",
                "Cannot link concepts across sentences semantically",
            ],
            requires_external=False,
        )

    def extract(
        self,
        encodings: List[TokenEncoding],
        source_text: str,
        **kwargs
    ) -> SemanticGraph:
        """
        Extract semantic graph from token encodings.

        Steps:
        1. Identify entity tokens (high SUBSTANTIVES or entity primitives)
        2. Group tokens by sentence
        3. Build nodes (one per unique lemma)
        4. Build edges from sentence structure (SVO) or co-occurrence
        """
        # Step 1: Identify entities
        entity_tokens = [t for t in encodings if t.is_entity()]

        # Step 2: Group by sentence
        by_sentence = self._group_by_sentence(encodings)

        # Step 3: Build nodes
        nodes = self._build_nodes(entity_tokens)

        # Step 4: Build edges
        edges = self._build_edges(by_sentence, nodes)

        return SemanticGraph(
            nodes=nodes,
            edges=edges,
            source_text=source_text,
            num_sentences=len(by_sentence),
        )

    def _group_by_sentence(
        self, encodings: List[TokenEncoding]
    ) -> Dict[int, List[TokenEncoding]]:
        """Group tokens by sentence index."""
        by_sent = {}
        for tok in encodings:
            if tok.sentence_idx not in by_sent:
                by_sent[tok.sentence_idx] = []
            by_sent[tok.sentence_idx].append(tok)
        return by_sent

    def _build_nodes(
        self, entity_tokens: List[TokenEncoding]
    ) -> Dict[str, GraphNode]:
        """
        Build graph nodes from entity tokens.

        Multiple tokens with same lemma -> single node with averaged vector.
        """
        nodes = {}

        for tok in entity_tokens:
            node_id = tok.lemma

            # Skip noise (very short lemmas)
            if len(node_id) < 2 and node_id.upper() not in ["I", "A"]:
                continue

            if node_id not in nodes:
                nodes[node_id] = GraphNode(
                    id=node_id,
                    label=tok.text,
                    vector=tok.vector.copy(),
                    mentions=[tok],
                    bank_activations=tok.bank_activations.copy(),
                )
            else:
                node = nodes[node_id]
                node.mentions.append(tok)
                # Running average for vector
                node.vector = (node.vector + tok.vector) / 2
                # Running average for banks
                for bank, val in tok.bank_activations.items():
                    old = node.bank_activations.get(bank, 0)
                    node.bank_activations[bank] = (old + val) / 2

        return nodes

    def _build_edges(
        self,
        by_sentence: Dict[int, List[TokenEncoding]],
        nodes: Dict[str, GraphNode]
    ) -> List[GraphEdge]:
        """
        Build edges from sentence structure.

        Within each sentence:
        - Find subject entities (SYN_SUBJ or position-based)
        - Find object entities (SYN_OBJ or position-based)
        - Find relation tokens (verbs, operators)
        - Connect subject -> object via relation
        """
        edges = []

        for sent_idx, tokens in by_sentence.items():
            # Get entities in this sentence that exist in our node set
            sentence_entities = [t for t in tokens if t.lemma in nodes]

            # Identify relations
            relations = [t for t in tokens if t.is_relation()]

            # Heuristic 1: Use explicit semantic roles if available
            subjects = [t for t in sentence_entities if "SYN_SUBJ" in t.sentence_primitives]
            objects = [t for t in sentence_entities if "SYN_OBJ" in t.sentence_primitives]

            # Heuristic 2: Fall back to SVO order
            if not subjects and not objects and len(sentence_entities) >= 2:
                subjects = [sentence_entities[0]]
                objects = sentence_entities[1:]

            # Create edges
            for subj in subjects:
                for obj in objects:
                    if subj.lemma == obj.lemma:
                        continue

                    rel = self._find_best_relation(subj, obj, relations, tokens)

                    rel_text = rel.text if rel else "related_to"
                    rel_vec = rel.vector if rel else np.zeros(8192)
                    rel_banks = rel.bank_activations if rel else {}

                    edge = GraphEdge(
                        source_id=subj.lemma,
                        target_id=obj.lemma,
                        relation_text=rel_text,
                        relation_vector=rel_vec,
                        relation_banks=rel_banks,
                        edge_type=GraphEdge.infer_type(rel_banks),
                        weight=1.0,
                        is_directed=True,
                    )
                    edges.append(edge)

            # Fallback: connect entities with weak co-occurrence edges
            if not edges and len(sentence_entities) > 1:
                edges.extend(self._build_proximity_edges(sentence_entities, tokens))

        return edges

    def _find_best_relation(
        self,
        subj: TokenEncoding,
        obj: TokenEncoding,
        relations: List[TokenEncoding],
        all_tokens: List[TokenEncoding]
    ) -> Optional[TokenEncoding]:
        """Find the relation token that best connects subject to object."""
        if not relations:
            return None

        # Return relation closest to midpoint between subject and object
        mid = (subj.token_idx + obj.token_idx) / 2
        best = min(relations, key=lambda r: abs(r.token_idx - mid))
        return best

    def _build_proximity_edges(
        self,
        entities: List[TokenEncoding],
        all_tokens: List[TokenEncoding]
    ) -> List[GraphEdge]:
        """Build weak edges between entities that co-occur in a sentence."""
        edges = []

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                if e1.lemma == e2.lemma:
                    continue

                # Find token between them as potential relation
                start = min(e1.token_idx, e2.token_idx)
                end = max(e1.token_idx, e2.token_idx)
                between = [t for t in all_tokens if start < t.token_idx < end]

                rel_text = "co_occurs"
                rel_vec = np.zeros(8192)
                rel_banks = {}

                if between:
                    mid = between[len(between) // 2]
                    rel_text = mid.lemma
                    rel_vec = mid.vector
                    rel_banks = mid.bank_activations

                edge = GraphEdge(
                    source_id=e1.lemma,
                    target_id=e2.lemma,
                    relation_text=rel_text,
                    relation_vector=rel_vec,
                    relation_banks=rel_banks,
                    edge_type=GraphEdge.infer_type(rel_banks),
                    weight=0.5,  # Weaker than explicit structure
                    is_directed=False,
                )
                edges.append(edge)

        return edges
