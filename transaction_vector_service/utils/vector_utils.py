"""
Vector utility functions.

This module provides utilities for vector operations,
similarity calculations, and embedding manipulations.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    
    # Convert to numpy arrays for better performance
    a = np.array(v1)
    b = np.array(v2)
    
    # Calculate cosine similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Euclidean distance
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return float('inf')
    
    # Convert to numpy arrays for better performance
    a = np.array(v1)
    b = np.array(v2)
    
    # Calculate Euclidean distance
    return np.linalg.norm(a - b)


def manhattan_distance(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the Manhattan distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Manhattan distance
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return float('inf')
    
    # Calculate Manhattan distance
    return sum(abs(a - b) for a, b in zip(v1, v2))


def normalize_vector(v: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Vector to normalize
        
    Returns:
        Normalized vector
    """
    if not v:
        return []
    
    # Convert to numpy array
    a = np.array(v)
    
    # Calculate norm
    norm = np.linalg.norm(a)
    
    # Avoid division by zero
    if norm == 0:
        return [0.0] * len(v)
    
    # Normalize and convert back to list
    return (a / norm).tolist()


def average_vectors(vectors: List[List[float]]) -> List[float]:
    """
    Calculate the average of multiple vectors.
    
    Args:
        vectors: List of vectors
        
    Returns:
        Average vector
    """
    if not vectors:
        return []
    
    # Ensure all vectors have the same dimension
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        raise ValueError("All vectors must have the same dimension")
    
    # Convert to numpy array
    arr = np.array(vectors)
    
    # Calculate mean and convert back to list
    return np.mean(arr, axis=0).tolist()


def weighted_average_vectors(vectors: List[List[float]], weights: List[float]) -> List[float]:
    """
    Calculate the weighted average of multiple vectors.
    
    Args:
        vectors: List of vectors
        weights: List of weights
        
    Returns:
        Weighted average vector
    """
    if not vectors or not weights or len(vectors) != len(weights):
        return []
    
    # Ensure all vectors have the same dimension
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        raise ValueError("All vectors must have the same dimension")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return [0.0] * dim
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Convert to numpy arrays
    arr = np.array(vectors)
    w = np.array(normalized_weights).reshape(-1, 1)
    
    # Calculate weighted average and convert back to list
    return (arr.T @ w).reshape(-1).tolist()


def vector_quantization(v: List[float], precision: int = 2) -> List[float]:
    """
    Quantize a vector to reduce storage size.
    
    Args:
        v: Vector to quantize
        precision: Number of decimal places
        
    Returns:
        Quantized vector
    """
    if not v:
        return []
    
    return [round(x, precision) for x in v]


def vector_to_bytes(v: List[float], precision: int = 2) -> bytes:
    """
    Convert a vector to bytes for storage.
    
    Args:
        v: Vector to convert
        precision: Number of decimal places for quantization
        
    Returns:
        Bytes representation
    """
    import struct
    
    if not v:
        return b''
    
    # Quantize vector
    quantized = vector_quantization(v, precision)
    
    # Convert to bytes using struct
    return struct.pack(f"{len(quantized)}f", *quantized)


def bytes_to_vector(b: bytes, vector_size: int) -> List[float]:
    """
    Convert bytes back to a vector.
    
    Args:
        b: Bytes to convert
        vector_size: Expected vector size
        
    Returns:
        Vector as list of floats
    """
    import struct
    
    if not b:
        return [0.0] * vector_size
    
    # Unpack bytes using struct
    return list(struct.unpack(f"{vector_size}f", b))


def find_nearest_vectors(
    query_vector: List[float],
    vector_db: List[Tuple[Any, List[float]]],
    top_k: int = 5,
    similarity_fn=cosine_similarity
) -> List[Tuple[Any, float]]:
    """
    Find the nearest vectors to a query vector.
    
    Args:
        query_vector: Query vector
        vector_db: List of (id, vector) tuples
        top_k: Number of nearest vectors to return
        similarity_fn: Similarity function to use
        
    Returns:
        List of (id, similarity) tuples for the top k nearest vectors
    """
    if not query_vector or not vector_db:
        return []
    
    # Calculate similarities
    similarities = [
        (item_id, similarity_fn(query_vector, vector))
        for item_id, vector in vector_db
    ]
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:top_k]


def compress_vector(v: List[float], dimensions: int) -> List[float]:
    """
    Compress a vector to a lower dimension.
    Very simple implementation using averaging of adjacent values.
    
    Args:
        v: Vector to compress
        dimensions: Target number of dimensions
        
    Returns:
        Compressed vector
    """
    if not v or dimensions >= len(v):
        return v
    
    # Convert to numpy array
    arr = np.array(v)
    
    # Simple compression by taking average of blocks
    block_size = len(v) // dimensions
    compressed = []
    
    for i in range(dimensions):
        start = i * block_size
        end = (i + 1) * block_size if i < dimensions - 1 else len(v)
        compressed.append(np.mean(arr[start:end]))
    
    return compressed