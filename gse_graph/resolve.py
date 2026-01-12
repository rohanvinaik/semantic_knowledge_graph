"""
Entity resolution via cosine clustering.

No LLM needed. Entities with similar vectors are the same entity.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from .structures import GraphNode, SemanticGraph


def resolve_entities(
    graph: SemanticGraph, 
    threshold: float = 0.15
) -> SemanticGraph:
    """
    Merge nodes that represent the same entity.
    
    Uses cosine similarity clustering on node vectors.
    Threshold of 0.15 means nodes with cosine distance < 0.15 merge.
    """
    if len(graph.nodes) < 2:
        return graph
    
    node_ids = list(graph.nodes.keys())
    # Stack vectors
    vectors = np.array([graph.nodes[nid].vector for nid in node_ids])
    
    # Normalize for cosine distance
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = vectors / norms
    
    # Calculate pairwise cosine distances
    # 1 - cosine_similarity = cosine_distance
    # pdist with 'cosine' returns cosine distance
    distances = pdist(normalized, metric='cosine')
    
    # Clustering
    # distance_threshold is used if n_clusters=None
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine', # Valid for scikit-learn >= 1.2
        linkage='average'
    )
    
    try:
        labels = clustering.fit_predict(normalized)
    except TypeError:
        # Fallback for older scikit-learn versions that might expect precomputed distance matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        dist_matrix = squareform(distances)
        labels = clustering.fit_predict(dist_matrix)
        
    # Group by cluster label
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node_ids[idx])
        
    # Create new merged nodes
    new_nodes = {}
    id_mapping = {}  # old_id -> new_id
    
    for label, cluster_ids in clusters.items():
        # Primary ID is the shortest lemma (most canonical usually) or most frequent
        # Let's pick shortest
        primary_id = min(cluster_ids, key=len)
        
        # Merge all into primary
        merged_node = graph.nodes[primary_id] # Start with primary
        
        for other_id in cluster_ids:
            id_mapping[other_id] = primary_id
            if other_id == primary_id:
                continue
                
            other_node = graph.nodes[other_id]
            
            # Weighted average of vectors (could weight by mentions count)
            n_merged = len(merged_node.mentions)
            n_other = len(other_node.mentions)
            total = n_merged + n_other
            
            merged_node.vector = (merged_node.vector * n_merged + other_node.vector * n_other) / total
            merged_node.mentions.extend(other_node.mentions)
            
            # Merge bank activations
            for bank, val in other_node.bank_activations.items():
                old = merged_node.bank_activations.get(bank, 0)
                merged_node.bank_activations[bank] = (old * n_merged + val * n_other) / total
                
        new_nodes[primary_id] = merged_node
        
    # Rebuild edges with new IDs
    new_edges = []
    for edge in graph.edges:
        src = id_mapping.get(edge.source_id)
        dst = id_mapping.get(edge.target_id)
        
        if src and dst and src != dst:
            # Update edge
            edge.source_id = src
            edge.target_id = dst
            new_edges.append(edge)
            
    # Deduplicate edges?
    # For now, keep all, visualizer can handle multi-edges
    
    graph.nodes = new_nodes
    graph.edges = new_edges
    
    return graph
