# GSE Semantic Knowledge Graph

A Python library for extracting knowledge graphs from text using **Grounded Semantic Encoding (GSE)**.

## Why This Is Different

Most knowledge graph extractors use one of two approaches:

| Approach | How It Works | Limitation |
|----------|--------------|------------|
| **Dependency Parsing** (spaCy, Stanford NLP) | Parse syntax tree → extract subject-verb-object triplets | Captures grammar, not meaning. "Bank" in "river bank" vs "savings bank" gets same treatment |
| **Embedding Similarity** (TransE, knowledge graph embeddings) | Encode entities as dense vectors, measure similarity | Opaque. Cannot explain *why* two concepts are related |
| **LLM Extraction** (GPT, Claude) | Prompt model to extract entities/relations | Expensive, slow, hallucinates, no grounding |

**GSE takes a fundamentally different approach:**

1. **Position IS meaning** - Text is encoded into 8192-dimensional sparse ternary vectors where each dimension corresponds to a semantic primitive (MENTAL, SPATIAL, TEMPORAL, CAUSAL, etc.)

2. **Multi-dimensional grounding** - Concepts are positioned in 5 hierarchical dimensions (SPATIAL, TEMPORAL, DOMAIN, TAXONOMIC, SCALE) derived from Wikipedia structure

3. **Anchor-based connectivity** - Entities connect through typed semantic anchors (SCOPE, HISTORY, KNOWN_FOR) enabling cross-domain linking without explicit edges

4. **Interpretable** - Every relationship can be traced: "Einstein connects to Physics via SCOPE anchor, to 20th century via TEMPORAL dimension"

## Architecture: Cassette System

The library uses a **cassette system** allowing you to swap extraction strategies:

```
┌─────────────────────────────────────────────────────────┐
│                    GSE Pipeline                         │
│                                                         │
│   Text → encode_text() → TokenEncodings                 │
│                              ↓                          │
│                    ┌─────────┴─────────┐                │
│                    ↓                   ↓                │
│            ┌──────────────┐    ┌──────────────┐         │
│            │   Grammar    │    │   Semantic   │         │
│            │   Cassette   │    │   Cassette   │         │
│            │              │    │              │         │
│            │ • SVO order  │    │ • Wiki       │         │
│            │ • Co-occur   │    │   grounding  │         │
│            │ • Bank types │    │ • Spreading  │         │
│            │              │    │   activation │         │
│            │ Fast: 0.3ms  │    │ • Anchors    │         │
│            └──────────────┘    │              │         │
│                                │ Rich: 500ms  │         │
│                                └──────────────┘         │
│                              ↓                          │
│                      SemanticGraph                      │
│                   (nodes, edges, 3D viz)                │
└─────────────────────────────────────────────────────────┘
```

### Grammar Cassette (`cassette="grammar"`)

**Fast, syntax-based extraction.**

- Extracts entities based on SUBSTANTIVES bank activation
- Creates edges from sentence structure (subject-verb-object)
- Types edges using bank activations (CAUSAL, TEMPORAL, SPATIAL)
- **Speed**: ~0.3ms per graph, ~19,000 edges/sec
- **Use case**: Quick structural analysis, grammar visualization

### Semantic Cassette (`cassette="semantic"`) [Default]

**Knowledge-grounded extraction using Wikipedia entity linking.**

- Grounds concepts to Wikipedia (10K+ vital articles)
- Multi-dimensional positioning (SPATIAL, TEMPORAL, DOMAIN, TAXONOMIC, SCALE)
- Spreading activation for semantic context retrieval
- Anchor-based cross-node connectivity (SCOPE, HISTORY, KNOWN_FOR)
- EPA coordinates (Evaluation/Potency/Activity) for affective semantics
- **Speed**: ~500ms per graph
- **Use case**: Concept understanding, knowledge synthesis, semantic similarity

## Installation

```bash
# Clone and install
git clone <repo>
cd semantic_knowledge_graph
pip install -e .

# For semantic cassette, also need sparse-wiki-grounding:
# Ensure ~/sparse-wiki-grounding exists with wiki_grounding.db
```

## Usage

### Basic Pipeline

```python
from gse_graph import encode_text, extract_graph, resolve_entities

text = """
Einstein discovered relativity. His work transformed physics
and our understanding of space and time.
"""

# 1. Encode text into semantic vectors
encodings = encode_text(text)

# 2. Extract graph (uses semantic cassette by default)
graph = extract_graph(encodings, text)

# 3. Resolve duplicate entities
graph = resolve_entities(graph)

# 4. Export for visualization
output = graph.to_json()
```

### Choosing a Cassette

```python
# Semantic grounding (default) - rich understanding
graph = extract_graph(encodings, text, cassette="semantic")

# Grammar/co-occurrence - fast structural analysis
graph = extract_graph(encodings, text, cassette="grammar")
```

### Direct Cassette Access

```python
from gse_graph.cassettes import get_cassette

# Get cassette with custom config
cassette = get_cassette("semantic", min_grounding_confidence=0.4)

# Extract
graph = cassette.extract(encodings, text)

# Benchmark
stats = cassette.benchmark(encodings, text, iterations=10)
print(f"Average time: {stats['avg_time_ms']:.1f}ms")
```

## Benchmarks

Tested on Apple M1 Pro:

| Metric | Grammar Cassette | Semantic Cassette |
|--------|------------------|-------------------|
| **Extraction Time** | ~0.3ms | ~500ms |
| **Tokens/sec** | 350,000+ | 200+ |
| **Edges/sec** | 19,000+ | 100+ |
| **Grounded Concepts** | 0% | ~50% |
| **Semantic Understanding** | Syntax only | Full meaning |

### When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Quick text structure analysis | Grammar |
| Real-time applications | Grammar |
| Concept relationship understanding | Semantic |
| Knowledge base construction | Semantic |
| Cross-domain linking | Semantic |
| Visualization with meaning | Semantic |

## Comparison to Other Systems

### vs. spaCy Named Entity Recognition

```python
# spaCy: "ADHD" → LABEL: ORG (wrong!)
# GSE: "ADHD" → Ungrounded but linked via co-occurrence to medical concepts

# spaCy: "Flowers for Algernon" → Multiple entities
# GSE: Single concept node with phrase detection
```

spaCy focuses on **entity classification** (PERSON, ORG, GPE). GSE focuses on **semantic decomposition** - what banks activate, what anchors connect.

### vs. Knowledge Graph Embeddings (TransE, RotatE)

```python
# KGE: entity_vector ≈ other_vector means "related"
# But WHY? No explanation.

# GSE: Einstein connects to Physics via:
#   - DOMAIN dimension: Science → Physics → Theoretical Physics
#   - SCOPE anchor: "Relativity"
#   - HISTORY anchor: "20th century"
#   - EPA: Evaluation=+1 (positive), Potency=+1 (strong), Activity=+1 (active)
```

KGE gives similarity scores. GSE gives **interpretable multi-dimensional positioning**.

### vs. LLM Extraction (GPT-4, Claude)

```python
# LLM: "Extract entities and relations from this text"
# → Expensive, slow, may hallucinate, no grounding to KB

# GSE: Deterministic extraction with Wikipedia grounding
# → Fast, consistent, verifiable
```

LLMs are powerful but expensive and non-deterministic. GSE provides **grounded, interpretable extraction** at a fraction of the cost.

## Semantic Banks

The system uses these semantic banks for typing:

| Bank | Detects | Example Triggers |
|------|---------|------------------|
| SUBSTANTIVES | Entities, nouns | people, objects, places |
| ACTION | Verbs, processes | create, destroy, move |
| MENTAL | Cognitive concepts | think, believe, feel |
| TEMPORAL | Time relations | before, after, during |
| SPATIAL | Location/position | above, inside, near |
| LOGICAL | Causal connections | because, therefore, if |
| EVALUATORS | Value judgments | good, bad, important |

## Output Format

```python
graph.to_json()
# Returns:
{
    "nodes": [
        {
            "id": "Q_Winston_Churchill",
            "label": "Winston Churchill",
            "x": 12.5, "y": -3.2, "z": 8.1,  # 3D position from dimensions
            "color": "rgb(200,180,220)",       # From EPA values
            "banks": {"MENTAL": 0.4, "TEMPORAL": 0.3},
            "mentions": 2
        },
        ...
    ],
    "edges": [
        {
            "source": "Q_Winston_Churchill",
            "target": "Q_British_Empire",
            "label": "shared:British Empire",
            "type": "MENTAL",
            "weight": 0.49,
            "directed": false
        },
        ...
    ],
    "metadata": {
        "num_nodes": 24,
        "num_edges": 55,
        "num_sentences": 5
    }
}
```

## Visualization

The JSON output is designed for `3d-force-graph` or similar libraries:

```javascript
import ForceGraph3D from '3d-force-graph';

const graph = ForceGraph3D()(document.getElementById('container'))
    .graphData(gseOutput)
    .nodeColor(n => n.color)
    .linkColor(l => edgeTypeColors[l.type]);
```

## Future Work

- **Extended grounding**: Integrate full Wikipedia (not just vital articles)
- **Literary knowledge base**: Add books, stories, fictional works
- **Real-time streaming**: Incremental graph updates
- **Query interface**: "How does X relate to Y?"

## Dependencies

- `semantic_probing` (local) - GSE encoding
- `sparse-wiki-grounding` (optional) - Wikipedia entity linking
- numpy, scipy, scikit-learn

## License

MIT
