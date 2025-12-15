"""
MUVERA (Multi-Vector Retrieval as Sparse Alignment) Implementation
Based on Google's implementation for efficient ColBERT-style retrieval.

This implementation converts multi-vector embeddings (like ColBERT) into 
fixed-dimensional encodings that can be efficiently searched.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EncodingType(Enum):
    """Type of encoding to use."""
    QUERY_SUM = "query_sum"  # Sum vectors in each partition (for queries)
    DOCUMENT_AVERAGE = "document_average"  # Average vectors in each partition (for documents)


class ProjectionType(Enum):
    """Type of projection to use."""
    IDENTITY = "identity"  # Use original dimensions
    AMS_SKETCH = "ams_sketch"  # Use sparse random projections


@dataclass
class MuveraConfig:
    """Configuration for MUVERA encoding."""
    dimension: int  # Dimension of input vectors (e.g., 128 for ColBERT)
    num_simhash_projections: int = 4  # Number of SimHash bits (creates 2^k partitions)
    projection_type: ProjectionType = ProjectionType.AMS_SKETCH
    projection_dimension: int = 32  # Dimension after projection
    num_repetitions: int = 1  # Number of hash repetitions for robustness
    seed: int = 42
    fill_empty_partitions: bool = True  # Fill empty partitions with nearest point
    final_projection_dimension: Optional[int] = None  # Optional final count sketch dimension


class MuveraEncoder:
    """
    MUVERA encoder that converts multi-vector embeddings to fixed-dimensional vectors.
    
    This enables efficient retrieval by:
    1. Partitioning token embeddings using SimHash
    2. Aggregating vectors within each partition
    3. Projecting to lower dimensions
    """
    
    def __init__(self, config: MuveraConfig):
        self.config = config
        self.num_partitions = 2 ** config.num_simhash_projections
        self._initialize_projection_matrices()
    
    def _initialize_projection_matrices(self):
        """Initialize random projection matrices."""
        np.random.seed(self.config.seed)
        
        # SimHash matrix (Gaussian random projections)
        if self.config.num_simhash_projections > 0:
            self.simhash_matrices = []
            for rep in range(self.config.num_repetitions):
                seed = self.config.seed + rep
                np.random.seed(seed)
                simhash_matrix = np.random.randn(
                    self.config.dimension, 
                    self.config.num_simhash_projections
                )
                self.simhash_matrices.append(simhash_matrix)
        
        # Projection matrix (sparse random projections for AMS sketch)
        if self.config.projection_type == ProjectionType.AMS_SKETCH:
            self.projection_matrices = []
            proj_dim = self.config.projection_dimension
            
            for rep in range(self.config.num_repetitions):
                seed = self.config.seed + rep
                np.random.seed(seed)
                
                # Sparse projection: each row has one non-zero entry (+1 or -1)
                projection_matrix = np.zeros((self.config.dimension, proj_dim))
                for i in range(self.config.dimension):
                    idx = np.random.randint(0, proj_dim)
                    sign = 2 * np.random.randint(0, 2) - 1  # +1 or -1
                    projection_matrix[i, idx] = sign
                
                self.projection_matrices.append(projection_matrix)
    
    def _simhash_partition_index(self, vector: np.ndarray) -> int:
        """
        Compute partition index using SimHash (Locality Sensitive Hashing).
        
        Args:
            vector: Projected vector from SimHash matrix
            
        Returns:
            Partition index (Gray code)
        """
        gray_code = 0
        for i in range(len(vector)):
            bit = 1 if vector[i] > 0 else 0
            # Append to Gray code
            gray_code = (gray_code << 1) + (bit ^ (gray_code & 1))
        return gray_code
    
    def _gray_code_to_binary(self, gray_code: int) -> int:
        """Convert Gray code to binary."""
        return gray_code ^ (gray_code >> 1)
    
    def _distance_to_partition(self, vector: np.ndarray, partition_index: int) -> int:
        """
        Compute Hamming distance from vector to a partition.
        
        Args:
            vector: Projected vector from SimHash
            partition_index: Target partition (Gray code)
            
        Returns:
            Hamming distance
        """
        distance = 0
        binary_rep = self._gray_code_to_binary(partition_index)
        
        for i in range(len(vector) - 1, -1, -1):
            cur_bit = 1 if vector[i] > 0 else 0
            distance += (cur_bit != (binary_rep & 1))
            binary_rep >>= 1
        
        return distance
    
    def _apply_count_sketch(self, vector: np.ndarray, final_dim: int) -> np.ndarray:
        """Apply Count Sketch for final dimensionality reduction."""
        out = np.zeros(final_dim)
        np.random.seed(self.config.seed)
        
        for i in range(len(vector)):
            index = np.random.randint(0, final_dim)
            sign = 2 * np.random.randint(0, 2) - 1
            out[index] += sign * vector[i]
        
        return out
    
    def encode_query(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Encode query multi-vector embedding using sum aggregation.
        
        Args:
            point_cloud: Shape (num_tokens, dimension) - e.g., (32, 128) for ColBERT query
            
        Returns:
            Fixed-dimensional encoding
        """
        num_points, dim = point_cloud.shape
        assert dim == self.config.dimension, f"Dimension mismatch: {dim} vs {self.config.dimension}"
        
        if self.config.projection_type == ProjectionType.IDENTITY:
            proj_dim = self.config.dimension
        else:
            proj_dim = self.config.projection_dimension
        
        output = np.zeros(self.config.num_repetitions * self.num_partitions * proj_dim)
        
        for rep in range(self.config.num_repetitions):
            # Apply SimHash projection
            if self.config.num_simhash_projections > 0:
                sketch = point_cloud @ self.simhash_matrices[rep]
            
            # Apply dimension reduction projection
            if self.config.projection_type == ProjectionType.IDENTITY:
                projected = point_cloud
            else:
                projected = point_cloud @ self.projection_matrices[rep]
            
            # Assign each point to its partition and sum
            for point_idx in range(num_points):
                if self.config.num_simhash_projections > 0:
                    partition_idx = self._simhash_partition_index(sketch[point_idx])
                else:
                    partition_idx = 0
                
                start_idx = rep * (self.num_partitions * proj_dim) + partition_idx * proj_dim
                end_idx = start_idx + proj_dim
                
                output[start_idx:end_idx] += projected[point_idx]
        
        # Optional final projection
        if self.config.final_projection_dimension is not None:
            output = self._apply_count_sketch(output, self.config.final_projection_dimension)
        
        return output
    
    def encode_document(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Encode document multi-vector embedding using average aggregation.
        
        Args:
            point_cloud: Shape (num_tokens, dimension) - e.g., (256, 128) for ColBERT document
            
        Returns:
            Fixed-dimensional encoding
        """
        num_points, dim = point_cloud.shape
        assert dim == self.config.dimension, f"Dimension mismatch: {dim} vs {self.config.dimension}"
        
        if self.config.projection_type == ProjectionType.IDENTITY:
            proj_dim = self.config.dimension
        else:
            proj_dim = self.config.projection_dimension
        
        output = np.zeros(self.config.num_repetitions * self.num_partitions * proj_dim)
        
        for rep in range(self.config.num_repetitions):
            # Apply SimHash projection
            if self.config.num_simhash_projections > 0:
                sketch = point_cloud @ self.simhash_matrices[rep]
            
            # Apply dimension reduction projection
            if self.config.projection_type == ProjectionType.IDENTITY:
                projected = point_cloud
            else:
                projected = point_cloud @ self.projection_matrices[rep]
            
            # Track partition sizes for averaging
            partition_sizes = np.zeros(self.num_partitions)
            
            # Assign each point to its partition and sum
            for point_idx in range(num_points):
                if self.config.num_simhash_projections > 0:
                    partition_idx = self._simhash_partition_index(sketch[point_idx])
                else:
                    partition_idx = 0
                
                start_idx = rep * (self.num_partitions * proj_dim) + partition_idx * proj_dim
                end_idx = start_idx + proj_dim
                
                output[start_idx:end_idx] += projected[point_idx]
                partition_sizes[partition_idx] += 1
            
            # Normalize by partition size (convert sum to average)
            for partition_idx in range(self.num_partitions):
                start_idx = rep * (self.num_partitions * proj_dim) + partition_idx * proj_dim
                end_idx = start_idx + proj_dim
                
                if partition_sizes[partition_idx] == 0 and self.config.num_simhash_projections > 0:
                    if self.config.fill_empty_partitions:
                        # Find nearest point to this partition
                        min_distance = float('inf')
                        nearest_point_idx = -1
                        
                        for point_idx in range(num_points):
                            distance = self._distance_to_partition(sketch[point_idx], partition_idx)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_point_idx = point_idx
                        
                        if nearest_point_idx >= 0:
                            output[start_idx:end_idx] = projected[nearest_point_idx]
                else:
                    # Average the vectors in this partition
                    if partition_sizes[partition_idx] > 0:
                        output[start_idx:end_idx] /= partition_sizes[partition_idx]
        
        # Optional final projection
        if self.config.final_projection_dimension is not None:
            output = self._apply_count_sketch(output, self.config.final_projection_dimension)
        
        return output
    
    def encode_batch_queries(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """Encode multiple queries."""
        return np.array([self.encode_query(pc) for pc in point_clouds])
    
    def encode_batch_documents(self, point_clouds: List[np.ndarray]) -> np.ndarray:
        """Encode multiple documents."""
        return np.array([self.encode_document(pc) for pc in point_clouds])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def batch_cosine_similarity(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and document encodings.
    
    Args:
        queries: Shape (num_queries, encoding_dim)
        documents: Shape (num_documents, encoding_dim)
        
    Returns:
        Similarity matrix of shape (num_queries, num_documents)
    """
    # Normalize
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    documents_norm = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    return queries_norm @ documents_norm.T


# Example usage and testing
if __name__ == "__main__":
    # Example: ColBERT-style embeddings
    print("MUVERA Encoding Example\n" + "=" * 50)
    
    # Configuration
    config = MuveraConfig(
        dimension=128,  # ColBERT embedding dimension
        num_simhash_projections=4,  # 2^4 = 16 partitions
        projection_dimension=32,  # Reduce to 32 dimensions
        num_repetitions=1,
        fill_empty_partitions=True,
    )
    
    encoder = MuveraEncoder(config)
    
    # Simulate ColBERT embeddings
    query_tokens = 32
    doc_tokens = 256
    
    query_embedding = np.random.randn(query_tokens, 128)  # Query: 32 tokens
    doc_embedding = np.random.randn(doc_tokens, 128)  # Document: 256 tokens
    
    # Encode
    query_encoded = encoder.encode_query(query_embedding)
    doc_encoded = encoder.encode_document(doc_embedding)
    
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Document embedding shape: {doc_embedding.shape}")
    print(f"\nQuery encoded shape: {query_encoded.shape}")
    print(f"Document encoded shape: {doc_encoded.shape}")
    
    # Compute similarity
    similarity = cosine_similarity(query_encoded, doc_encoded)
    print(f"\nCosine similarity: {similarity:.4f}")
    
    # Batch example
    print("\n" + "=" * 50)
    print("Batch Encoding Example")
    
    num_queries = 100
    num_docs = 1000
    
    query_batch = [np.random.randn(32, 128) for _ in range(num_queries)]
    doc_batch = [np.random.randn(256, 128) for _ in range(num_docs)]
    
    queries_encoded = encoder.encode_batch_queries(query_batch)
    docs_encoded = encoder.encode_batch_documents(doc_batch)
    
    print(f"\nEncoded {num_queries} queries: {queries_encoded.shape}")
    print(f"Encoded {num_docs} documents: {docs_encoded.shape}")
    
    # Compute all similarities
    similarities = batch_cosine_similarity(queries_encoded, docs_encoded)
    print(f"Similarity matrix shape: {similarities.shape}")
    
    # Get top-k for each query
    k = 10
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    print(f"\nTop-{k} documents for first query: {top_k_indices[0]}")
    
    print("\n" + "=" * 50)
    print("Compression ratio:")
    original_size = query_tokens * 128 + doc_tokens * 128
    encoded_size = len(query_encoded) + len(doc_encoded)
    print(f"Original: {original_size} floats")
    print(f"Encoded: {encoded_size} floats")
    print(f"Compression: {original_size / encoded_size:.2f}x")

