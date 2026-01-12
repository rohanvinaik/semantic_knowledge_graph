"""
Semantic Grounding Cassette.

Extracts graph structure based on:
1. Entity grounding to Wikipedia knowledge base
2. Spreading activation for semantic context
3. Anchor-based cross-node connectivity
4. Multi-dimensional positioning (SPATIAL, TEMPORAL, DOMAIN, etc.)

This is a knowledge-grounded extraction that understands actual
concept meanings and creates semantically meaningful relationships.

Use cases:
- Concept relationship analysis
- Knowledge graph construction
- Semantic similarity visualization
- Cross-domain concept linking
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from .base import ExtractionCassette, CassetteInfo
from ..structures import TokenEncoding, GraphNode, GraphEdge, SemanticGraph

# Add sparse-wiki-grounding to path
SPARSE_WIKI_PATH = Path.home() / "sparse-wiki-grounding" / "src"
if SPARSE_WIKI_PATH.exists():
    sys.path.insert(0, str(SPARSE_WIKI_PATH))

try:
    from wiki_grounding import (
        EntityStore,
        EntityProfile,
        SpreadingActivation,
        SpreadingConfig,
        ActivationResult,
        SemanticBank,
        ANCHOR_TO_BANK,
        GroundingDimension,
    )
    _HAS_WIKI_GROUNDING = True
except ImportError:
    _HAS_WIKI_GROUNDING = False


# Default database path - prefer larger relational-ai database if available
RELATIONAL_AI_DB = Path.home() / "relational-ai" / "data" / "sparse_wiki.db"
SPARSE_WIKI_DB = Path.home() / "sparse-wiki-grounding" / "data" / "wiki_grounding.db"
DEFAULT_DB_PATH = RELATIONAL_AI_DB if RELATIONAL_AI_DB.exists() else SPARSE_WIKI_DB


@dataclass
class GroundedConcept:
    """A concept from text grounded to Wikipedia."""
    text: str                        # Original text
    lemma: str                       # Normalized form
    entity_id: Optional[str]         # Wikipedia Q-number if grounded
    entity_label: Optional[str]      # Wikipedia label
    entity_description: Optional[str]  # Wikipedia description
    confidence: float                # Grounding confidence (0-1)
    positions: Dict[str, Any]        # Multi-dimensional positions
    anchors: List[Tuple[str, str, float]]  # (label, category, weight)
    source_tokens: List[TokenEncoding]  # Original token encodings
    vector: np.ndarray               # Combined GSE vector


class SemanticGroundingCassette(ExtractionCassette):
    """
    Knowledge-grounded graph extraction using sparse-wiki entity linking.

    This cassette:
    1. Identifies potential concepts in text (nouns, proper nouns, key terms)
    2. Grounds them to Wikipedia entities with multi-dimensional positions
    3. Uses spreading activation to find semantic relationships
    4. Creates edges based on shared anchors and semantic similarity

    Strengths:
    - Understands actual concept meanings
    - Links concepts across sentences semantically
    - Multi-dimensional positioning (SPATIAL, TEMPORAL, DOMAIN)
    - Anchor-based cross-domain connectivity

    Weaknesses:
    - Slower than grammar-based extraction
    - Requires external database (~31MB)
    - Limited to concepts in Wikipedia (10K+ vital articles)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        min_grounding_confidence: float = 0.3,
        spreading_decay: float = 0.6,
        spreading_threshold: float = 0.15,
        use_anchor_spreading: bool = True,
    ):
        """
        Initialize the semantic grounding cassette.

        Args:
            db_path: Path to wiki_grounding.db (default: ~/sparse-wiki-grounding/data/)
            min_grounding_confidence: Minimum confidence to accept a grounding
            spreading_decay: Decay factor for spreading activation
            spreading_threshold: Minimum activation threshold
            use_anchor_spreading: Enable cross-node anchor spreading
        """
        if not _HAS_WIKI_GROUNDING:
            raise ImportError(
                "SemanticGroundingCassette requires sparse-wiki-grounding. "
                "Ensure ~/sparse-wiki-grounding exists with wiki_grounding.db"
            )

        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(f"Wiki grounding database not found: {self.db_path}")

        self.min_confidence = min_grounding_confidence
        self.spreading_config = SpreadingConfig(
            decay=spreading_decay,
            threshold=spreading_threshold,
            use_anchors=use_anchor_spreading,
            max_depth=2,
            max_results=100,
        )

        self._store: Optional[EntityStore] = None
        self._spreader: Optional[SpreadingActivation] = None

    @property
    def store(self) -> EntityStore:
        """Lazy-load the entity store."""
        if self._store is None:
            self._store = EntityStore(self.db_path)
        return self._store

    @property
    def spreader(self) -> SpreadingActivation:
        """Lazy-load the spreading activation system."""
        if self._spreader is None:
            self._spreader = SpreadingActivation(self.store, self.spreading_config)
        return self._spreader

    @property
    def info(self) -> CassetteInfo:
        return CassetteInfo(
            name="semantic_grounding",
            description="Knowledge-grounded extraction using Wikipedia entity linking and spreading activation",
            strengths=[
                "Understands actual concept meanings",
                "Links concepts semantically across sentences",
                "Multi-dimensional positioning (SPATIAL, TEMPORAL, DOMAIN)",
                "Anchor-based cross-domain connectivity",
                "EPA semantic coordinates (Evaluation, Potency, Activity)",
                "Interpretable typed relationships",
            ],
            weaknesses=[
                "Slower than grammar-based extraction",
                "Requires external database (~31MB)",
                "Limited to Wikipedia vital articles coverage",
            ],
            requires_external=True,
            external_dependency="sparse-wiki-grounding (wiki_grounding.db)",
        )

    def extract(
        self,
        encodings: List[TokenEncoding],
        source_text: str,
        **kwargs
    ) -> SemanticGraph:
        """
        Extract a semantically-grounded graph from encoded tokens.

        Steps:
        1. Identify candidate concepts (entities + key terms)
        2. Ground concepts to Wikipedia entities
        3. Build nodes with grounded semantic information
        4. Build edges using spreading activation and anchor overlap
        """
        # Step 1: Identify candidate concepts
        candidates = self._identify_candidates(encodings, source_text)

        # Step 2: Ground to Wikipedia
        grounded = self._ground_concepts(candidates)

        # Step 3: Build nodes
        nodes = self._build_grounded_nodes(grounded, encodings)

        # Step 4: Build semantically-meaningful edges
        edges = self._build_semantic_edges(grounded, nodes)

        # Step 5: Compute 3D positions from multi-dimensional grounding
        self._compute_positions(nodes)

        # Count sentences
        num_sentences = len(set(t.sentence_idx for t in encodings))

        return SemanticGraph(
            nodes=nodes,
            edges=edges,
            source_text=source_text,
            num_sentences=num_sentences,
        )

    def _identify_candidates(
        self,
        encodings: List[TokenEncoding],
        source_text: str
    ) -> List[Dict[str, Any]]:
        """
        Identify candidate concepts to ground.

        Looks for:
        - Explicit entities (high SUBSTANTIVES bank)
        - Proper nouns (capitalized words not at sentence start)
        - Multi-word proper noun sequences (e.g., "Patrick Winston", "New York")
        - Multi-word phrases in quotes
        - Key domain terms
        """
        candidates = []
        seen_lemmas: Set[str] = set()

        # Group tokens by sentence for context
        by_sentence: Dict[int, List[TokenEncoding]] = {}
        for tok in encodings:
            if tok.sentence_idx not in by_sentence:
                by_sentence[tok.sentence_idx] = []
            by_sentence[tok.sentence_idx].append(tok)

        # =====================================================================
        # FIRST: Find multi-word proper noun sequences (e.g., "Patrick Winston")
        # =====================================================================
        for sent_idx, sent_tokens in by_sentence.items():
            i = 0
            while i < len(sent_tokens):
                tok = sent_tokens[i]

                # Check if this starts a proper noun sequence
                if tok.text and tok.text[0].isupper() and not tok.text.isupper():
                    # Collect consecutive capitalized words
                    sequence = [tok]
                    j = i + 1
                    while j < len(sent_tokens):
                        next_tok = sent_tokens[j]
                        # Continue if capitalized and adjacent (no major gap)
                        if (next_tok.text and
                            next_tok.text[0].isupper() and
                            not next_tok.text.isupper() and
                            next_tok.token_idx == sequence[-1].token_idx + 1):
                            sequence.append(next_tok)
                            j += 1
                        else:
                            break

                    # If we have a multi-word sequence, add it as a phrase candidate
                    if len(sequence) >= 2:
                        phrase_text = " ".join(t.text for t in sequence)
                        phrase_lemma = phrase_text.lower()

                        if phrase_lemma not in seen_lemmas:
                            seen_lemmas.add(phrase_lemma)
                            # Also mark individual words as seen to avoid duplicates
                            for t in sequence:
                                seen_lemmas.add(t.lemma)

                            candidates.append({
                                "text": phrase_text,
                                "lemma": phrase_lemma,
                                "tokens": sequence,
                                "is_phrase": True,
                                "is_proper_noun_sequence": True,
                                "banks": {},
                            })
                            i = j
                            continue

                i += 1

        # =====================================================================
        # SECOND: Find single-word entities (with strict filtering)
        # =====================================================================
        # Common function words, pronouns, and verbs that should NEVER be entity candidates
        STOP_WORDS = {
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'who', 'what', 'which', 'whom', 'whose',
            # Determiners
            'the', 'a', 'an', 'some', 'any', 'no', 'every', 'each', 'all', 'both',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
            # Conjunctions
            'and', 'or', 'but', 'if', 'because', 'although', 'while', 'since', 'unless',
            # Common verbs that shouldn't be entities
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'said', 'says', 'say', 'make', 'makes', 'made',
            'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew',
            'think', 'thought', 'take', 'took', 'give', 'gave', 'find', 'found',
            # Other common words
            'not', 'just', 'only', 'also', 'very', 'too', 'so', 'then', 'now', 'here',
            'there', 'when', 'where', 'how', 'why', 'more', 'most', 'other', 'another',
            'such', 'own', 'same', 'different', 'new', 'old', 'good', 'bad', 'first',
            'last', 'long', 'little', 'much', 'many', 'few', 'less', 'view', 'way',
            'something', 'anything', 'nothing', 'everything', 'someone', 'anyone',
            'everyone', 'similar', 'resonates', 'believed', 'believes', 'construct',
        }

        # Track all occurrences per lemma for accurate mention counting
        lemma_to_candidate: Dict[str, Dict[str, Any]] = {}

        for tok in encodings:
            # Skip stop words
            if tok.lemma.lower() in STOP_WORDS or tok.text.lower() in STOP_WORDS:
                continue

            is_candidate = False

            # High SUBSTANTIVES activation (nouns)
            if tok.bank_activations.get("SUBSTANTIVES", 0) > 0.003:
                is_candidate = True

            # Entity primitives
            if tok.word_primitives & {"SOMEONE", "SOMETHING", "PEOPLE"}:
                is_candidate = True

            # Proper noun heuristic (capitalized, not at sentence start, not all caps)
            if tok.text and tok.text[0].isupper() and tok.token_idx > 0 and not tok.text.isupper():
                is_candidate = True

            # Skip very short terms (likely noise)
            if len(tok.lemma) < 3:
                continue

            if is_candidate:
                if tok.lemma in lemma_to_candidate:
                    # Add this occurrence to existing candidate
                    lemma_to_candidate[tok.lemma]["all_tokens"].append(tok)
                else:
                    # Create new candidate entry
                    lemma_to_candidate[tok.lemma] = {
                        "text": tok.text,
                        "lemma": tok.lemma,
                        "tokens": [tok],  # First occurrence for grounding
                        "all_tokens": [tok],  # All occurrences for mention counting
                        "is_entity": tok.is_entity(),
                        "banks": tok.bank_activations,
                    }
                seen_lemmas.add(tok.lemma)

        # Add accumulated candidates
        candidates.extend(lemma_to_candidate.values())

        # =====================================================================
        # THIRD: Find quoted phrases (potential titles, works, etc.)
        # =====================================================================
        quoted = re.findall(r'"([^"]+)"', source_text)
        for phrase in quoted:
            if len(phrase) > 2 and phrase.lower() not in seen_lemmas:
                # Find character position of phrase in source
                phrase_start = source_text.find(f'"{phrase}"')
                if phrase_start == -1:
                    continue
                phrase_end = phrase_start + len(phrase) + 2  # +2 for quotes

                # Find tokens within this character range (approximate)
                # Use first few words of phrase to find representative tokens
                phrase_words = phrase.split()[:5]  # First 5 words
                phrase_tokens = [t for t in encodings
                                if t.text in phrase_words and t.text.lower() not in STOP_WORDS]

                if phrase_tokens:
                    # For quotes, all_tokens = single representative token (1 mention)
                    candidates.append({
                        "text": phrase,
                        "lemma": phrase.lower(),
                        "tokens": phrase_tokens[:3],  # First few for grounding
                        "all_tokens": phrase_tokens[:1],  # Single token = 1 mention
                        "is_phrase": True,
                        "banks": {},
                    })
                    seen_lemmas.add(phrase.lower())

        return candidates

    def _ground_concepts(
        self, candidates: List[Dict[str, Any]]
    ) -> List[GroundedConcept]:
        """
        Ground candidate concepts to Wikipedia entities using context-aware disambiguation.

        Two-pass approach that lets disambiguation emerge from the architecture:
        1. First pass: Ground unambiguous candidates (exact matches, single results)
        2. Build context: Spread activation from grounded entities
        3. Second pass: Disambiguate ambiguous candidates using context activation

        This allows "Winston" to resolve to Patrick Winston (AI) vs Winston Churchill
        based on surrounding context like "storytelling", "intelligence", "AI".
        """
        grounded = []
        ambiguous: List[Tuple[Dict[str, Any], List[EntityProfile]]] = []
        grounded_entity_ids: Set[str] = set()

        # =====================================================================
        # FIRST PASS: Ground unambiguous candidates
        # =====================================================================
        for cand in candidates:
            search_term = cand["text"]

            # Try exact match first, then fuzzy
            # Use higher limit to ensure we capture all relevant candidates for disambiguation
            results = self.store.search_exact(search_term, limit=10)
            if not results:
                results = self.store.search(search_term, limit=20)

            if not results:
                # No matches - will be ungrounded
                grounded.append(self._make_ungrounded_concept(cand))
                continue

            # Check if unambiguous (single high-confidence match or exact label)
            exact_matches = [r for r in results if r.entity.label.lower() == search_term.lower()]

            if len(exact_matches) == 1:
                # Single exact match - unambiguous
                profile = exact_matches[0]
                grounded.append(self._make_grounded_concept(cand, profile, 0.9))
                grounded_entity_ids.add(profile.entity.id)
            elif len(results) == 1:
                # Single result - unambiguous
                profile = results[0]
                best = self._score_grounding(cand, [profile])
                if best and best[1] >= self.min_confidence:
                    grounded.append(self._make_grounded_concept(cand, best[0], best[1]))
                    grounded_entity_ids.add(best[0].entity.id)
                else:
                    grounded.append(self._make_ungrounded_concept(cand))
            else:
                # Multiple candidates - defer to context-aware disambiguation
                ambiguous.append((cand, results))

        # =====================================================================
        # BUILD CONTEXT: Spread activation from grounded entities
        # =====================================================================
        context_activation: Dict[str, float] = {}
        context_anchors: Set[str] = set()

        if grounded_entity_ids and ambiguous:
            # Spread from all grounded entities
            sources = {eid: 1.0 for eid in grounded_entity_ids}
            activated = self.spreader.spread_multiple(sources, use_anchors=True)

            for result in activated:
                eid = result.entity.entity.id
                context_activation[eid] = result.activation

            # Collect anchors from grounded concepts for similarity matching
            for concept in grounded:
                if concept.entity_id:
                    for anchor_label, _, _ in concept.anchors:
                        context_anchors.add(anchor_label.lower())

        # =====================================================================
        # BUILD CONTEXT DECOMPOSITION: Multi-layer semantic primitives
        # =====================================================================
        # Layer 0: Direct context anchors from grounded entities
        # Layer 1: Anchors of those anchors (recursive decomposition)
        # Layer 2: One more level deep

        context_layers: List[Set[str]] = [context_anchors]  # Layer 0

        # Build deeper layers by decomposing anchors
        for depth in range(2):  # Build layers 1 and 2
            prev_layer = context_layers[-1]
            next_layer: Set[str] = set()

            for anchor_label in list(prev_layer)[:30]:  # Limit for performance
                # Search for entity matching this anchor
                anchor_results = self.store.search_exact(anchor_label, limit=1)
                if anchor_results:
                    anchor_entity = anchor_results[0]
                    # Get this entity's anchors (decomposition)
                    sub_anchors = self.store.get_entity_anchors(anchor_entity.entity.id)
                    for sa in sub_anchors[:10]:
                        next_layer.add(sa[1].lower())

            context_layers.append(next_layer)

        # =====================================================================
        # SECOND PASS: Recursive disambiguation with confidence trajectory
        # =====================================================================
        for cand, results in ambiguous:
            best_profile = None
            best_score = -float('inf')
            best_trajectory = []

            for profile in results:
                # Base score from label matching
                base_result = self._score_grounding(cand, [profile])
                if not base_result:
                    continue
                _, base_score = base_result

                eid = profile.entity.id

                # =========================================================
                # RECURSIVE SIMILARITY: Track confidence at each layer
                # =========================================================
                trajectory = []  # Confidence at each decomposition layer

                # Get candidate's anchor decomposition layers
                candidate_layers: List[Set[str]] = []

                # Layer 0: Direct anchors
                entity_anchors = self.store.get_entity_anchors(eid)
                layer0 = {a[1].lower() for a in entity_anchors[:20]}
                candidate_layers.append(layer0)

                # Build deeper layers for candidate
                for depth in range(2):
                    prev = candidate_layers[-1]
                    next_layer: Set[str] = set()
                    for anchor_label in list(prev)[:20]:
                        anchor_results = self.store.search_exact(anchor_label, limit=1)
                        if anchor_results:
                            sub_anchors = self.store.get_entity_anchors(anchor_results[0].entity.id)
                            for sa in sub_anchors[:8]:
                                next_layer.add(sa[1].lower())
                    candidate_layers.append(next_layer)

                # Compute overlap at each layer and track trajectory
                for layer_idx in range(min(len(context_layers), len(candidate_layers))):
                    ctx_layer = context_layers[layer_idx]
                    cand_layer = candidate_layers[layer_idx]

                    if len(ctx_layer) > 0 and len(cand_layer) > 0:
                        overlap = ctx_layer & cand_layer
                        # Jaccard-like similarity
                        union_size = len(ctx_layer | cand_layer)
                        similarity = len(overlap) / union_size if union_size > 0 else 0
                        trajectory.append(similarity)
                    else:
                        trajectory.append(0.0)

                # =========================================================
                # TRAJECTORY ANALYSIS: Convergence vs Divergence
                # =========================================================
                # Compute raw context alignment score from trajectory
                trajectory_score = 0.0

                if len(trajectory) >= 2:
                    # Trajectory delta: positive = converging, negative = diverging
                    for i in range(1, len(trajectory)):
                        delta = trajectory[i] - trajectory[i-1]
                        layer_weight = 1.0 + (i * 0.5)
                        trajectory_score += delta * layer_weight

                    # Absolute overlap at each layer
                    for i, sim in enumerate(trajectory):
                        layer_weight = 1.0 + (i * 0.3)
                        trajectory_score += sim * layer_weight

                # Penalize entities with no anchors (can't compute trajectory)
                if len(candidate_layers[0]) == 0:
                    trajectory_score -= 0.5

                # Description keyword matching
                desc_score = 0.0
                if profile.entity.description:
                    desc_lower = profile.entity.description.lower()
                    for layer_idx, ctx_layer in enumerate(context_layers):
                        matches = sum(1 for anchor in ctx_layer if anchor in desc_lower)
                        desc_score += matches * 0.15 * (1 + layer_idx * 0.2)

                # =========================================================
                # DYNAMIC WEIGHTING: Uncertainty-scaled trajectory influence
                # =========================================================
                # When base_score is low (ambiguous), trajectory matters MORE
                # When base_score is high (clear match), trust popularity
                #
                # uncertainty = how unsure we are based on label matching alone
                # trajectory_influence scales UP with uncertainty

                uncertainty = 1.0 - min(base_score, 1.0)  # 0 = certain, 1 = uncertain

                # Trajectory influence increases with uncertainty
                # At high uncertainty (0.5+), trajectory can dominate
                # At low uncertainty (<0.3), base score dominates
                trajectory_influence = 0.3 + (uncertainty * 0.7)  # Range: 0.3 to 1.0

                # Normalize trajectory_score to similar scale as base_score
                normalized_trajectory = trajectory_score * 0.4

                # Final score: blend based on uncertainty
                # More uncertain = more weight on trajectory
                total_score = (
                    base_score * (1.0 - trajectory_influence * 0.5) +
                    normalized_trajectory * trajectory_influence +
                    desc_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_profile = profile
                    best_trajectory = trajectory

            if best_profile and best_score >= self.min_confidence:
                grounded.append(self._make_grounded_concept(cand, best_profile, min(max(best_score, 0), 1.0)))
                grounded_entity_ids.add(best_profile.entity.id)
            else:
                grounded.append(self._make_ungrounded_concept(cand))

        return grounded

    def _make_grounded_concept(
        self,
        cand: Dict[str, Any],
        profile: EntityProfile,
        confidence: float
    ) -> GroundedConcept:
        """Create a grounded concept from candidate and entity profile."""
        # Extract positions
        positions = {}
        for pos in profile.positions:
            dim_name = pos.dimension.value if hasattr(pos.dimension, 'value') else str(pos.dimension)
            positions[dim_name] = {
                "depth": pos.path_depth,
                "sign": pos.path_sign,
                "path": pos.path_nodes,
            }

        # Extract anchors
        anchors = self.store.get_entity_anchors(profile.entity.id)
        anchor_list = [(a[1], a[2], a[3]) for a in anchors[:20]]

        # Combine GSE vectors from source tokens
        vectors = [t.vector for t in cand["tokens"]]
        combined_vec = np.mean(vectors, axis=0) if vectors else np.zeros(8192)

        return GroundedConcept(
            text=cand["text"],
            lemma=cand["lemma"],
            entity_id=profile.entity.id,
            entity_label=profile.entity.label,
            entity_description=profile.entity.description,
            confidence=confidence,
            positions=positions,
            anchors=anchor_list,
            source_tokens=cand.get("all_tokens", cand["tokens"]),  # Use all occurrences
            vector=combined_vec,
        )

    def _make_ungrounded_concept(self, cand: Dict[str, Any]) -> GroundedConcept:
        """Create an ungrounded concept from candidate."""
        vectors = [t.vector for t in cand["tokens"]]
        combined_vec = np.mean(vectors, axis=0) if vectors else np.zeros(8192)

        return GroundedConcept(
            text=cand["text"],
            lemma=cand["lemma"],
            entity_id=None,
            entity_label=None,
            entity_description=None,
            confidence=0.0,
            positions={},
            anchors=[],
            source_tokens=cand.get("all_tokens", cand["tokens"]),  # Use all occurrences
            vector=combined_vec,
        )

    def _score_grounding(
        self,
        candidate: Dict[str, Any],
        results: List[EntityProfile]
    ) -> Optional[Tuple[EntityProfile, float]]:
        """
        Score grounding candidates and return best match with confidence.

        Scoring factors:
        - Label similarity (exact match, prefix, contains)
        - Entity importance (vital level, pagerank)
        - Semantic compatibility (bank alignment)
        - Length ratio penalty (avoid matching short words to long entities)
        """
        if not results:
            return None

        best_profile = None
        best_score = 0.0

        cand_text = candidate["text"].lower().strip()
        cand_banks = candidate.get("banks", {})
        is_phrase = candidate.get("is_phrase", False)

        # Skip very short common words unless they're phrases
        if len(cand_text) <= 3 and not is_phrase:
            if cand_text in {"the", "a", "an", "is", "are", "was", "were", "be",
                           "it", "i", "he", "she", "we", "they", "what", "who",
                           "how", "why", "when", "this", "that", "for", "to"}:
                return None

        for profile in results:
            score = 0.0
            label = profile.entity.label.lower()

            # Length ratio check - penalize if candidate is much shorter than entity
            len_ratio = len(cand_text) / max(len(label), 1)
            if len_ratio < 0.3 and not is_phrase:
                continue  # Skip if candidate is <30% of entity length

            # Label matching - stricter scoring
            if label == cand_text:
                score += 0.6  # Exact match
            elif cand_text == label.split()[0] and len(cand_text) > 3:
                score += 0.45  # First word match (e.g., "Winston" -> "Winston Churchill")
            elif label.startswith(cand_text) and len(cand_text) > 4:
                score += 0.35  # Prefix match
            elif cand_text in label and len(cand_text) > 5:
                score += 0.2  # Substring match (only if long enough)
            else:
                # Fuzzy match - much lower score
                score += 0.05

            # For phrases (quoted text), require stronger match
            if is_phrase:
                # Must be nearly exact for phrases
                if label != cand_text and cand_text not in label:
                    score *= 0.3

            # Importance bonus (smaller than before)
            if profile.entity.vital_level:
                importance = max(0, 1 - profile.entity.vital_level / 10)
                score += importance * 0.1

            if profile.entity.pagerank:
                score += min(profile.entity.pagerank * 0.5, 0.1)

            # Proper noun bonus - if candidate is capitalized
            if candidate["text"] and candidate["text"][0].isupper():
                if profile.entity.label and profile.entity.label[0].isupper():
                    score += 0.1

            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile and best_score >= self.min_confidence:
            return (best_profile, min(best_score, 1.0))
        return None

    def _build_grounded_nodes(
        self,
        concepts: List[GroundedConcept],
        encodings: List[TokenEncoding]
    ) -> Dict[str, GraphNode]:
        """
        Build graph nodes from grounded concepts.

        Nodes include:
        - Grounded entity information
        - Multi-dimensional positions
        - Semantic anchors for edge building
        """
        nodes = {}

        for concept in concepts:
            # Use entity_id if grounded, else lemma
            node_id = concept.entity_id or concept.lemma

            # Label: prefer entity label, fall back to text
            label = concept.entity_label or concept.text

            # Build bank activations from anchors
            bank_activations = {}
            for anchor_label, category, weight in concept.anchors:
                bank = ANCHOR_TO_BANK.get(category, SemanticBank.MENTAL)
                bank_name = bank.value if hasattr(bank, 'value') else str(bank)
                bank_activations[bank_name] = bank_activations.get(bank_name, 0) + weight

            # Merge with GSE bank activations from source tokens
            for tok in concept.source_tokens:
                for bank, val in tok.bank_activations.items():
                    bank_activations[bank] = max(bank_activations.get(bank, 0), val)

            # For mention counting:
            # - source_tokens from all_tokens = each token is a separate occurrence
            # - source_tokens from tokens (phrases) = all tokens are ONE occurrence
            # We use unique sentence indices to count true occurrences
            if concept.source_tokens:
                seen_sentences = set()
                occurrence_tokens = []
                for tok in concept.source_tokens:
                    if tok.sentence_idx not in seen_sentences:
                        seen_sentences.add(tok.sentence_idx)
                        occurrence_tokens.append(tok)
                mentions_to_add = occurrence_tokens if occurrence_tokens else concept.source_tokens[:1]
            else:
                mentions_to_add = []

            if node_id in nodes:
                # Merge with existing node (same entity appearing multiple times)
                existing = nodes[node_id]
                existing.mentions.extend(mentions_to_add)
                # Average vectors
                existing.vector = (existing.vector + concept.vector) / 2
                # Max bank activations
                for bank, val in bank_activations.items():
                    existing.bank_activations[bank] = max(
                        existing.bank_activations.get(bank, 0), val
                    )
                # Merge anchors (keep unique)
                if concept.anchors:
                    existing_anchor_labels = {a[0] for a in (existing.wiki_anchors or [])}
                    for anchor in concept.anchors:
                        if anchor[0] not in existing_anchor_labels:
                            existing.wiki_anchors.append(anchor)
            else:
                nodes[node_id] = GraphNode(
                    id=node_id,
                    label=label,
                    vector=concept.vector,
                    mentions=mentions_to_add,
                    bank_activations=bank_activations,
                    wiki_entity_id=concept.entity_id,
                    wiki_anchors=concept.anchors,
                )

        return nodes

    def _build_semantic_edges(
        self,
        concepts: List[GroundedConcept],
        nodes: Dict[str, GraphNode]
    ) -> List[GraphEdge]:
        """
        Build edges based on semantic relationships.

        Edge sources:
        1. Spreading activation through entity links
        2. Shared semantic anchors
        3. Co-occurrence in same sentence (weak edges)
        """
        edges = []
        edge_set: Set[Tuple[str, str]] = set()

        # Get grounded entity IDs for spreading
        grounded_ids = {c.entity_id for c in concepts if c.entity_id}

        # 1. Spreading activation edges
        for concept in concepts:
            if not concept.entity_id:
                continue

            source_id = concept.entity_id

            # Spread from this entity
            activated = self.spreader.spread(source_id, initial_activation=1.0)

            for result in activated[:20]:
                target_id = result.entity.entity.id

                # Only create edge if target is in our concept set
                if target_id not in nodes:
                    continue

                edge_key = tuple(sorted([source_id, target_id]))
                if edge_key in edge_set:
                    continue
                edge_set.add(edge_key)

                # Determine edge type from relations and banks
                edge_type = self._infer_edge_type(result)

                # Relation text from path
                rel_text = result.relations[0] if result.relations else "semantically_related"
                if rel_text.startswith("anchor:"):
                    rel_text = rel_text[7:]  # Remove prefix

                edges.append(GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_text=rel_text,
                    relation_vector=np.zeros(8192),  # Could compute from relation
                    relation_banks={b.value: v for b, v in result.bank_activations.items()},
                    edge_type=edge_type,
                    weight=result.activation,
                    is_directed=True,
                    direction=self._infer_direction(result.relations),
                ))

        # 2. Shared anchor edges (for unconnected grounded concepts)
        for i, c1 in enumerate(concepts):
            if not c1.entity_id:
                continue

            for c2 in concepts[i + 1:]:
                if not c2.entity_id:
                    continue

                edge_key = tuple(sorted([c1.entity_id, c2.entity_id]))
                if edge_key in edge_set:
                    continue

                # Check for shared anchors
                c1_anchors = {a[0] for a in c1.anchors}
                c2_anchors = {a[0] for a in c2.anchors}
                shared = c1_anchors & c2_anchors

                if shared:
                    edge_set.add(edge_key)

                    # Find strongest shared anchor
                    shared_weights = []
                    for a1 in c1.anchors:
                        if a1[0] in shared:
                            for a2 in c2.anchors:
                                if a2[0] == a1[0]:
                                    shared_weights.append((a1[0], a1[1], a1[2] * a2[2]))

                    if shared_weights:
                        best = max(shared_weights, key=lambda x: x[2])

                        edges.append(GraphEdge(
                            source_id=c1.entity_id,
                            target_id=c2.entity_id,
                            relation_text=f"shared:{best[0]}",
                            relation_vector=np.zeros(8192),
                            relation_banks={},
                            edge_type=self._anchor_to_edge_type(best[1]),
                            weight=best[2],
                            is_directed=False,
                        ))

        # 3. Co-occurrence edges for ungrounded concepts
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                id1 = c1.entity_id or c1.lemma
                id2 = c2.entity_id or c2.lemma

                if id1 not in nodes or id2 not in nodes:
                    continue

                edge_key = tuple(sorted([id1, id2]))
                if edge_key in edge_set:
                    continue

                # Check if they co-occur in same sentence
                sents1 = {t.sentence_idx for t in c1.source_tokens}
                sents2 = {t.sentence_idx for t in c2.source_tokens}

                if sents1 & sents2:
                    edge_set.add(edge_key)
                    edges.append(GraphEdge(
                        source_id=id1,
                        target_id=id2,
                        relation_text="co_occurs",
                        relation_vector=np.zeros(8192),
                        relation_banks={},
                        edge_type="CONTEXTUAL",
                        weight=0.3,
                        is_directed=False,
                    ))

        return edges

    def _infer_edge_type(self, result: ActivationResult) -> str:
        """Infer edge type from spreading activation result."""
        # Check bank activations
        if result.bank_activations:
            dominant = max(result.bank_activations.items(), key=lambda x: x[1])
            bank = dominant[0]
            if hasattr(bank, 'value'):
                bank = bank.value
            return str(bank)

        # Check relation path
        if result.relations:
            rel = result.relations[0].lower()
            if any(x in rel for x in ["capital", "located", "place", "geography"]):
                return "SPATIAL"
            if any(x in rel for x in ["born", "died", "year", "history", "century"]):
                return "TEMPORAL"
            if any(x in rel for x in ["created", "wrote", "invented", "founded"]):
                return "CAUSAL"
            if any(x in rel for x in ["instance", "subclass", "type"]):
                return "TAXONOMIC"

        return "SEMANTIC"

    def _anchor_to_edge_type(self, category: Optional[str]) -> str:
        """Convert anchor category to edge type."""
        mapping = {
            "SCOPE": "MENTAL",
            "HISTORY": "TEMPORAL",
            "KNOWN_FOR": "SEMANTIC",
            "GEOGRAPHY": "SPATIAL",
            "TYPE": "TAXONOMIC",
        }
        return mapping.get(category, "SEMANTIC")

    def _infer_direction(self, relations: List[str]) -> Optional[str]:
        """Infer edge direction from relation types."""
        if not relations:
            return None

        rel = relations[0].lower()
        if "cause" in rel or "created" in rel or "founded" in rel:
            return "CAUSE_TO_EFFECT"
        if "before" in rel:
            return "BEFORE_TO_AFTER"
        if "part_of" in rel:
            return "PART_TO_WHOLE"
        return None

    def _compute_positions(self, nodes: Dict[str, GraphNode]):
        """
        Compute 3D positions from multi-dimensional grounding.

        Uses dimensional positions to place nodes in 3D space:
        - X: DOMAIN dimension (field of knowledge)
        - Y: TEMPORAL dimension (historical placement)
        - Z: SPATIAL/TAXONOMIC (geographic or categorical)
        """
        for node_id, node in nodes.items():
            if not node.wiki_entity_id:
                # Ungrounded: position based on bank activations
                x = node.bank_activations.get("MENTAL", 0) * 100
                y = node.bank_activations.get("TEMPORAL", 0) * 100
                z = node.bank_activations.get("SPATIAL", 0) * 100
                node.position_3d = np.array([x, y, z])
                continue

            # Get entity positions from store
            profile = self.store.get(node.wiki_entity_id)
            if not profile:
                node.position_3d = np.array([0.0, 0.0, 0.0])
                continue

            # Extract dimensional depths
            x = y = z = 0.0
            for pos in profile.positions:
                dim_name = pos.dimension.value if hasattr(pos.dimension, 'value') else str(pos.dimension)
                depth = pos.path_depth * pos.path_sign

                if dim_name == "DOMAIN":
                    x = depth * 10
                elif dim_name == "TEMPORAL":
                    y = depth * 10
                elif dim_name in ("SPATIAL", "TAXONOMIC"):
                    z = depth * 10

            # Add some spread based on pagerank
            if profile.entity.pagerank:
                spread = profile.entity.pagerank * 5
                x += np.random.uniform(-spread, spread)
                y += np.random.uniform(-spread, spread)
                z += np.random.uniform(-spread, spread)

            node.position_3d = np.array([x, y, z])

            # Color from EPA values
            if profile.epa:
                r = int(128 + (profile.epa.evaluation.value if profile.epa.evaluation else 0) * 127)
                g = int(128 + (profile.epa.potency.value if profile.epa.potency else 0) * 127)
                b = int(128 + (profile.epa.activity.value if profile.epa.activity else 0) * 127)
                node.color_rgb = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
