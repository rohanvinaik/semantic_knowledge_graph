from gse_graph import encode_text, extract_graph, resolve_entities
import json

def analyze_breadth():
    text = (
        "The project succeeded because the team worked hard. "
        "However, the budget was exceeded. "
        "Alice feels happy about the result."
    )
    
    print(f"Text: {text}\n")
    
    encodings = encode_text(text)
    graph = extract_graph(encodings, text)
    graph = resolve_entities(graph)
    
    print(f"Captured {len(graph.nodes)} nodes and {len(graph.edges)} edges.\n")
    
    print("--- Edges with Properties ---")
    for edge in graph.edges:
        print(f"Edge: {edge.source_id} --[{edge.relation_text}]--> {edge.target_id}")
        print(f"  Type: {edge.edge_type}")
        print(f"  Banks: {json.dumps(edge.relation_banks, indent=2)}")
        print("")
        
    print("--- Nodes with Semantic Properties ---")
    for node_id, node in graph.nodes.items():
        print(f"Node: {node.label}")
        # Show top banks
        top_banks = sorted(node.bank_activations.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top Dimensions: {top_banks}")

if __name__ == "__main__":
    analyze_breadth()
