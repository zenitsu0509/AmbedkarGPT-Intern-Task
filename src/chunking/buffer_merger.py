"""
Buffer Merger - Preserves contextual continuity in semantic chunks.

This module provides utilities for merging buffers and handling
overlapping context between chunks.
"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class BufferMerger:
    """
    Handles buffer merging operations for semantic chunks.
    
    Buffer merging ensures that chunks maintain contextual continuity
    by including overlapping context from adjacent chunks.
    """
    
    def __init__(
        self,
        buffer_size: int = 3,
        overlap_ratio: float = 0.2,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the buffer merger.
        
        Args:
            buffer_size: Number of sentences to include in buffer
            overlap_ratio: Ratio of overlap between chunks
            embedding_model: Pre-loaded embedding model (optional)
        """
        self.buffer_size = buffer_size
        self.overlap_ratio = overlap_ratio
        self.embedding_model = embedding_model
    
    def merge_adjacent_buffers(
        self,
        chunks: List[Dict],
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Merge adjacent chunks if they are highly similar.
        
        This helps consolidate chunks that were split but are
        semantically very close.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        current = chunks[0].copy()
        
        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            
            # Check similarity between current and next chunk
            if self.embedding_model and "embedding" in current and "embedding" in next_chunk:
                current_emb = np.array(current["embedding"]).reshape(1, -1)
                next_emb = np.array(next_chunk["embedding"]).reshape(1, -1)
                
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(current_emb, next_emb)[0][0]
                
                # Merge if highly similar and combined size is acceptable
                combined_tokens = current.get("token_count", 0) + next_chunk.get("token_count", 0)
                
                if sim >= similarity_threshold and combined_tokens <= 1024:
                    # Merge chunks
                    current = self._merge_two_chunks(current, next_chunk)
                    continue
            
            merged.append(current)
            current = next_chunk.copy()
        
        merged.append(current)
        return merged
    
    def _merge_two_chunks(self, chunk1: Dict, chunk2: Dict) -> Dict:
        """Merge two chunks into one."""
        merged_text = chunk1["text"] + " " + chunk2["text"]
        
        # Combine sentences if available
        sentences1 = chunk1.get("sentences", [chunk1["text"]])
        sentences2 = chunk2.get("sentences", [chunk2["text"]])
        merged_sentences = sentences1 + sentences2
        
        # Combine indices if available
        indices1 = chunk1.get("sentence_indices", [])
        indices2 = chunk2.get("sentence_indices", [])
        merged_indices = indices1 + indices2
        
        # Update embedding (average of both)
        if "embedding" in chunk1 and "embedding" in chunk2:
            emb1 = np.array(chunk1["embedding"])
            emb2 = np.array(chunk2["embedding"])
            merged_embedding = ((emb1 + emb2) / 2).tolist()
        else:
            merged_embedding = chunk1.get("embedding", [])
        
        return {
            "id": chunk1["id"],  # Keep first chunk's ID
            "text": merged_text,
            "sentences": merged_sentences,
            "sentence_indices": merged_indices,
            "buffer_before": chunk1.get("buffer_before", []),
            "buffer_after": chunk2.get("buffer_after", []),
            "full_context": chunk1.get("buffer_before", []) + [merged_text] + chunk2.get("buffer_after", []),
            "token_count": chunk1.get("token_count", 0) + chunk2.get("token_count", 0),
            "embedding": merged_embedding
        }
    
    def add_cross_chunk_context(
        self,
        chunks: List[Dict],
        context_sentences: int = 2
    ) -> List[Dict]:
        """
        Add cross-chunk context by including sentences from neighboring chunks.
        
        This creates richer context for each chunk by including
        ending sentences from the previous chunk and starting sentences
        from the next chunk.
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced = chunk.copy()
            
            # Add context from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_sentences = prev_chunk.get("sentences", [prev_chunk["text"]])
                enhanced["context_before"] = prev_sentences[-context_sentences:]
            else:
                enhanced["context_before"] = []
            
            # Add context from next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                next_sentences = next_chunk.get("sentences", [next_chunk["text"]])
                enhanced["context_after"] = next_sentences[:context_sentences]
            else:
                enhanced["context_after"] = []
            
            # Create expanded context
            all_context = (
                enhanced["context_before"] +
                enhanced.get("sentences", [enhanced["text"]]) +
                enhanced["context_after"]
            )
            enhanced["expanded_context"] = " ".join(all_context)
            
            enhanced_chunks.append(enhanced)
        
        return enhanced_chunks
    
    def create_sliding_window_chunks(
        self,
        sentences: List[str],
        window_size: int = 5,
        stride: int = 3,
        embeddings: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Create chunks using a sliding window approach.
        
        Alternative chunking method that creates overlapping windows
        of sentences.
        
        Args:
            sentences: List of sentences
            window_size: Number of sentences per window
            stride: Number of sentences to move window
            embeddings: Pre-computed sentence embeddings
        """
        chunks = []
        chunk_id = 0
        
        for start in range(0, len(sentences), stride):
            end = min(start + window_size, len(sentences))
            
            window_sentences = sentences[start:end]
            window_text = " ".join(window_sentences)
            
            # Calculate embedding if available
            if embeddings is not None:
                window_embedding = np.mean(embeddings[start:end], axis=0).tolist()
            else:
                window_embedding = []
            
            chunk = {
                "id": f"window_{chunk_id}",
                "text": window_text,
                "sentences": window_sentences,
                "sentence_indices": list(range(start, end)),
                "start_idx": start,
                "end_idx": end,
                "embedding": window_embedding
            }
            chunks.append(chunk)
            chunk_id += 1
            
            # Stop if we've reached the end
            if end >= len(sentences):
                break
        
        return chunks
    
    def deduplicate_chunks(
        self,
        chunks: List[Dict],
        similarity_threshold: float = 0.95
    ) -> List[Dict]:
        """
        Remove near-duplicate chunks based on embedding similarity.
        """
        if len(chunks) <= 1:
            return chunks
        
        deduplicated = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_duplicate = False
            
            if "embedding" in chunk:
                chunk_emb = np.array(chunk["embedding"]).reshape(1, -1)
                
                for existing in deduplicated:
                    if "embedding" in existing:
                        existing_emb = np.array(existing["embedding"]).reshape(1, -1)
                        
                        from sklearn.metrics.pairwise import cosine_similarity
                        sim = cosine_similarity(chunk_emb, existing_emb)[0][0]
                        
                        if sim >= similarity_threshold:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                deduplicated.append(chunk)
        
        return deduplicated
