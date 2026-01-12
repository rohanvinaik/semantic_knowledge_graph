"""
Semantic encoding wrapper.

This module wraps semantic_probing to encode text into vectors with
full bank activation and primitive decomposition.
"""

from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import re

# Import from the specific submodules where they are defined
from semantic_probing.encoding.sparse_ternary import (
    HadamardBasis, 
    SentenceScaleBasis, 
    bind, 
    bundle,
    D,  # Dimension
)
from semantic_probing.encoding.text_encoder import POS_MAP
from .structures import TokenEncoding

# Initialize encoder (singleton)
_encoder: Optional["TextEncoder"] = None

def get_encoder() -> "TextEncoder":
    global _encoder
    from semantic_probing.encoding.text_encoder import TextEncoder
    if _encoder is None:
        # Use default DB path from the library
        _encoder = TextEncoder()
    return _encoder

def encode_text(text: str) -> List[TokenEncoding]:
    """
    Encode text into list of TokenEncodings.
    """
    encoder = get_encoder()
    
    results = []
    # Use encoder's spacy nlp for consistency
    doc = encoder.nlp(text)
    
    # Track sentence index based on spacy's sent detection
    for sent_idx, sent in enumerate(doc.sents):
        tokens = [t for t in sent]
        
        for tok_idx, token in enumerate(tokens):
            if token.is_space:
                continue
                
            # Use TextEncoder's internal method to get the vector
            # This handles DB lookup, augmented lexicon, etc.
            # We call _encode_token which returns a SparseVector
            word_vec_sparse = encoder._encode_token(token)
            
            # If not found (None) or empty, use OOV encoding or zero?
            # TextEncoder.encode() skips them. We should probably keep them but flag them?
            # For graph, we want meaningful words.
            if word_vec_sparse is None or word_vec_sparse.nnz == 0:
                # Try OOV encoding so we have SOMETHING
                # Or just skip?
                # Let's use OOV fallback to allow "cat" even if not in DB (unlikely for cat, but still)
                word_vec_sparse = encode_oov(token.lemma_.lower(), encoder.word_basis)
                word_prims = set()
            else:
                 # Recover primitives from DB for metadata
                 # This is a bit inefficient (double lookup) but safe
                 word_prims = set()
                 decomp = encoder._lookup_word(token.lemma_.lower(), POS_MAP.get(token.pos_, 'n'))
                 if decomp:
                     word_prims = {p for p, v in decomp}
            
            word_vec = word_vec_sparse.to_dense().astype(np.float32)
            
            # Sentence-scale role
            # We can use dependency parsing from spacy directly!
            # DEP_TO_SYNTACTIC is in text_encoder.py
            from semantic_probing.encoding.text_encoder import DEP_TO_SYNTACTIC
            
            sent_prims = set()
            syn_role = DEP_TO_SYNTACTIC.get(token.dep_)
            if syn_role:
                sent_prims.add(syn_role)
                
            # Compose
            roles = set() # inferred high level roles
            if syn_role == "SYN_SUBJ":
                roles.add("agent")
            elif syn_role == "SYN_OBJ":
                roles.add("patient")

            # Compose word + sentence scale
            if sent_prims:
                role_vec_sparse = get_role_vector(sent_prims, encoder.sent_basis)
                # Use bundle() not bind() because word and sentence bases are orthogonal
                # bind() would result in zero vector
                composed_sparse = bundle([word_vec_sparse, role_vec_sparse])
                composed = composed_sparse.to_dense().astype(np.float32)
            else:
                composed = word_vec
            
            # Compute bank activations
            banks = compute_bank_activations(composed, encoder.word_basis)
            
            results.append(TokenEncoding(
                text=token.text,
                lemma=token.lemma_,
                vector=composed,
                word_primitives=word_prims,
                sentence_primitives=sent_prims,
                roles=roles,
                bank_activations=banks,
                sentence_idx=sent_idx,
                token_idx=tok_idx,
            ))
    
    return results



def compute_bank_activations(vec: np.ndarray, basis: HadamardBasis) -> Dict[str, float]:
    """
    Compute activation level for each semantic bank.
    
    Banks are 1024-dim slices of the 8192-dim vector.
    Activation = normalized dot product with bank's primitives.
    """
    banks = {}
    bank_names = [
        "SUBSTANTIVES", "QUANTITY", "EVALUATORS", "MENTAL",
        "ACTION", "TEMPORAL", "SPATIAL", "LOGICAL"
    ]
    
    # 8192 total dims / 8 banks = 1024 dims per bank
    # Assuming standard layout where dims are contiguous
    dims_per_bank = 1024
    
    for i, name in enumerate(bank_names):
        start = i * dims_per_bank
        end = start + dims_per_bank
        
        if start >= len(vec):
            break
            
        bank_slice = vec[start:end]
        
        # Activation = proportion of non-zero values (intensity)
        # Using L1 norm of the slice normalized by size
        activation = np.sum(np.abs(bank_slice)) / dims_per_bank
        banks[name] = float(activation)
    
    return banks


def encode_oov(word: str, basis: HadamardBasis) -> "SparseVector":
    """Encode out-of-vocabulary word via character n-grams."""
    # Bundle 3-grams
    if len(word) < 3:
        ngrams = [word]
    else:
        ngrams = [word[i:i+3] for i in range(len(word)-2)]
    
    from semantic_probing.encoding.sparse_ternary import SparseVector
    
    vecs = []
    for ng in ngrams:
        # Hash n-gram to basis index
        h = hash(ng)
        idx = h % basis.dimension
        polarity = 1 if (h // basis.dimension) % 2 == 0 else -1
        
        v = SparseVector(basis.dimension, [], [])
        if polarity > 0:
            v.positive_indices = np.array([idx], dtype=np.uint16)
        else:
            v.negative_indices = np.array([idx], dtype=np.uint16)
        vecs.append(v)
    
    return bundle(vecs) if vecs else SparseVector.zeros(basis.dimension)


def get_role_vector(sent_prims: Set[str], basis: SentenceScaleBasis) -> "SparseVector":
    """Get vector for sentence-scale primitives."""
    vecs = []
    for prim in sent_prims:
        v = basis.get_primitive(prim)
        if v is not None:
            vecs.append(v)
    
    if vecs:
        return bundle(vecs)
    
    from semantic_probing.encoding.sparse_ternary import SparseVector
    return SparseVector.zeros(basis.dimension)
