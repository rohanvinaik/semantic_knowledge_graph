import time
import sys
import numpy as np
from gse_graph import encode_text, extract_graph, resolve_entities

def generate_text(num_sentences=100):
    """Generate dummy text for benchmarking."""
    # Using some repetitive but structured text
    sentences = [
        "The cat sat on the mat.",
        "A dog chased the cat.",
        "The quick brown fox jumps over the lazy dog.",
        "Alice loves Bob because he is kind.",
        "Bob likes Alice but hates the cat.",
        "The sun rises in the east.",
        "Water is wet and fire is hot.",
        "Knowledge is power.",
        "To be or not to be.",
        "I think therefore I am."
    ]
    return " ".join(sentences * (num_sentences // 10))

def run_benchmark():
    text = generate_text(num_sentences=1000)
    print(f"Benchmarking with text length: {len(text)} chars")
    
    # Warmup
    print("Warming up...")
    _ = encode_text("Test sentence.")
    
    # 1. Encoding
    print("\n--- Encoding ---")
    start_time = time.time()
    encodings = encode_text(text)
    end_time = time.time()
    
    duration = end_time - start_time
    num_tokens = len(encodings)
    print(f"Time: {duration:.4f}s")
    print(f"Tokens: {num_tokens}")
    print(f"Speed: {num_tokens / duration:.2f} tokens/sec")
    
    # 2. Extraction
    print("\n--- Graph Extraction ---")
    start_time = time.time()
    graph = extract_graph(encodings, text)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Time: {duration:.4f}s")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Speed: {len(graph.edges) / duration:.2f} edges/sec")
    
    # 3. Resolution
    print("\n--- Entity Resolution ---")
    start_time = time.time()
    resolved_graph = resolve_entities(graph)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Time: {duration:.4f}s")
    print(f"Nodes after resolution: {len(resolved_graph.nodes)}")
    
if __name__ == "__main__":
    run_benchmark()
