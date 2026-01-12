import json
from gse_graph import encode_text, extract_graph, resolve_entities

def main():
    text = "The quick brown fox jumps over the lazy dog. The dog chased the cat."
    print("Input Text:", text)
    
    print("Encoding...")
    encodings = encode_text(text)
    
    print("Extracting Graph...")
    graph = extract_graph(encodings, text)
    
    print("Resolving Entities...")
    graph = resolve_entities(graph)
    
    output = graph.to_json()
    
    # Save to file
    with open("graph_output.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Graph JSON saved to graph_output.json")
    print(f"Nodes: {len(output['nodes'])}")
    print(f"Edges: {len(output['edges'])}")

if __name__ == "__main__":
    main()
