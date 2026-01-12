# CLAUDE.md - Project Context for Claude Code

## Project Overview

**Semantic Knowledge Graph** - A Python library for extracting knowledge graphs from text using Semantic Encoding. Unlike syntax-based extractors, this captures semantic intent and roles directly from vector space where "position IS meaning".

## Architecture (v0.2.0 - Cassette System)

```
gse_graph/
├── structures.py    # Core dataclasses: TokenEncoding, GraphNode, GraphEdge, SemanticGraph
├── encode.py        # Wraps semantic_probing.TextEncoder for 8192-dim sparse vectors
├── extract.py       # Main entry point - routes to cassettes
├── resolve.py       # Entity resolution via Agglomerative Clustering
└── cassettes/
    ├── __init__.py              # get_cassette() factory function
    ├── base.py                  # ExtractionCassette abstract base class
    ├── grammar_cooccurrence.py  # Fast syntax-based extraction
    └── semantic_grounding.py    # Knowledge-grounded extraction (sparse-wiki)
```

### Cassette System

Two extraction strategies available:

1. **Grammar Cassette** (`cassette="grammar"`)
   - Fast (~0.3ms): Syntax-based SVO extraction
   - Uses bank activations for edge typing
   - No semantic understanding of OOV concepts

2. **Semantic Cassette** (`cassette="semantic"`) [DEFAULT]
   - Rich (~500ms): Wikipedia entity grounding
   - Multi-dimensional positioning (SPATIAL, TEMPORAL, DOMAIN, etc.)
   - **Recursive disambiguation with trajectory tracking**
   - Anchor-based cross-node connectivity
   - Database: `~/relational-ai/data/sparse_wiki.db` (250K entities)

## Key Technical Details

### Vector Properties
- **Dimensions**: 8192-dimensional sparse ternary vectors
- **Density**: ~0.5% (approximately 40 non-zero values per vector)
- **Activation threshold**: `0.003` - CRITICAL for entity/relation detection
  - Previous failure: threshold of 0.3 produced zero results due to sparse nature

### Semantic Banks
Entity and relation typing uses these semantic banks:
- `SUBSTANTIVES`: Entities (nouns, concrete things)
- `ACTION`: Relations, verbs
- `LOGICAL`: Causal connections (because, therefore)
- `TEMPORAL`: Time relations (before, after)
- `SPATIAL`: Location/position
- `MENTAL`: Cognitive/emotional
- `EVALUATORS`: Value judgments (good, bad)

### Critical Implementation Notes
- **Use `bundle()` NOT `bind()`**: Combining word + sentence vectors must use additive bundling, not multiplicative binding. Using `bind()` produces zero vectors due to orthogonality.
- **OOV handling**: N-gram hashing for out-of-vocabulary words
- **Entity detection**: Check `SUBSTANTIVES` bank > 0.003 OR primitives contain {SOMEONE, SOMETHING, PEOPLE}
- **Relation detection**: Check `ACTION` bank > 0.003 OR sentence primitives contain {SYN_VERB}

## Dependencies

- **semantic_probing** (local): `/Users/rohanvinaik/semantic_probing` - provides TextEncoder, lexicon database (130k+ words)
- numpy, scipy, scikit-learn

## Benchmarks (M1 Pro)

| Metric | Speed |
|--------|-------|
| Encoding | ~3,800 tokens/sec |
| Extraction | ~19,000 edges/sec |

## Development Commands

```bash
# Install in dev mode
pip install -e .

# Run benchmarks
python benchmarks/benchmark_speed.py
python benchmarks/benchmark_breadth.py

# Run demo
python examples/demo.py
```

## Pipeline Usage

```python
from gse_graph import encode_text, extract_graph, resolve_entities

text = "The project succeeded because the team worked hard."
encodings = encode_text(text)
graph = extract_graph(encodings, text)
graph = resolve_entities(graph)
output = graph.to_json()  # For 3d-force-graph visualization
```

## Related Projects

- `semantic_probing` - Core encoding library (`~/semantic_probing`)
- `sparse-wiki-grounding` - Entity linking to Wikipedia (`~/sparse-wiki-grounding`)
  - Database: `~/sparse-wiki-grounding/data/wiki_grounding.db` (31MB, 10K entities)
  - Provides: EntityStore, SpreadingActivation, anchor layer
- `relational-ai` - Higher-level cognitive architecture using Semantic Encoding

## Cassette Usage

```python
# Default: semantic grounding
graph = extract_graph(encodings, text)

# Explicit cassette selection
graph = extract_graph(encodings, text, cassette="grammar")
graph = extract_graph(encodings, text, cassette="semantic")

# Direct cassette access
from gse_graph.cassettes import get_cassette
cassette = get_cassette("semantic", min_grounding_confidence=0.4)
graph = cassette.extract(encodings, text)
```

## Recursive Disambiguation Architecture

The semantic cassette uses **recursive semantic decomposition with trajectory tracking** for context-aware disambiguation:

### The Problem
When "Winston" appears in text about "storytelling" and "cognition", should it resolve to:
- Winston Churchill (higher popularity/pagerank)
- Patrick Winston (AI researcher, wrote about storytelling and cognition)

### The Solution: Multi-Layer Trajectory Analysis

1. **Layer 0**: Direct anchors from grounded context entities
   - "Storytelling" → [Entertainment, Folklore, Literature, Writing]
   - "Computation" → [Arithmetic, Computer, Algorithm, Brain]

2. **Layer 1+**: Recursive decomposition of anchors
   - Decompose each anchor into ITS anchors
   - Build semantic primitive tree 2-3 layers deep

3. **Trajectory Tracking**: At each layer, compute overlap between:
   - Candidate entity's decomposition
   - Context's decomposition
   - Track if overlap INCREASES (converging) or DECREASES (diverging)

4. **Dynamic Weighting**:
   ```python
   uncertainty = 1.0 - base_score  # Low confidence = high uncertainty
   trajectory_influence = 0.3 + (uncertainty * 0.7)  # Range: 0.3 to 1.0
   ```
   - **High uncertainty** → Trajectory can override popularity
   - **Low uncertainty** → Trust base score (exact matches)

### Result
Patrick Winston wins because:
- Layer 0: "Computer science" overlaps with "Computation"
- Layer 1: Deeper decomposition shows convergence with cognitive/AI context
- Trajectory: CONVERGING (+0.132) vs Churchill's (+0.086)
- Despite lower pagerank, better semantic alignment wins

## Known Limitations (Semantic Cassette)

- Database has ~250K entities (vital articles + DBpedia)
- Literary works (e.g., "Flowers for Algernon") may not be in database
- Grounding confidence tuning may be needed for specific domains
