"""
Result Ranker - Combine and rank results from local and global search.

Implements re-ranking and deduplication of retrieved results.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ResultRanker:
    """
    Combine and rank results from local and global RAG searches.
    
    Provides unified ranking, deduplication, and relevance scoring.
    """
    
    def __init__(
        self,
        local_weight: float = 0.6,
        global_weight: float = 0.4,
        diversity_penalty: float = 0.1
    ):
        """
        Initialize the ranker.
        
        Args:
            local_weight: Weight for local search scores
            global_weight: Weight for global search scores
            diversity_penalty: Penalty for similar results (0-1)
        """
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.diversity_penalty = diversity_penalty
    
    def combine_results(
        self,
        local_results: List[Dict],
        global_results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Combine and rank results from both search methods.
        
        Args:
            local_results: Results from local graph RAG search
            global_results: Results from global graph RAG search
            top_k: Number of final results to return
            
        Returns:
            Combined and ranked list of results
        """
        # Create unified scoring
        combined = {}
        
        # Process local results
        for result in local_results:
            chunk_id = result.get("id")
            if chunk_id not in combined:
                combined[chunk_id] = result.copy()
                combined[chunk_id]["local_score"] = result.get("combined_score", 0)
                combined[chunk_id]["global_score"] = 0
                combined[chunk_id]["source"] = "local"
            else:
                combined[chunk_id]["local_score"] = result.get("combined_score", 0)
        
        # Process global results
        for result in global_results:
            chunk_id = result.get("id")
            if chunk_id not in combined:
                combined[chunk_id] = result.copy()
                combined[chunk_id]["global_score"] = result.get("global_score", 0)
                combined[chunk_id]["local_score"] = 0
                combined[chunk_id]["source"] = "global"
            else:
                combined[chunk_id]["global_score"] = result.get("global_score", 0)
                combined[chunk_id]["source"] = "both"
        
        # Calculate final scores
        results_list = []
        for chunk_id, result in combined.items():
            local_score = result.get("local_score", 0)
            global_score = result.get("global_score", 0)
            
            # Weighted combination
            final_score = (
                self.local_weight * local_score +
                self.global_weight * global_score
            )
            
            # Bonus for appearing in both searches
            if result.get("source") == "both":
                final_score *= 1.2
            
            result["final_score"] = final_score
            results_list.append(result)
        
        # Sort by final score
        results_list.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Apply diversity re-ranking
        diverse_results = self._apply_diversity(results_list, top_k)
        
        return diverse_results
    
    def _apply_diversity(
        self,
        results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Apply diversity penalty to avoid redundant results.
        
        Uses a greedy approach to select diverse results.
        """
        if len(results) <= top_k:
            return results
        
        selected = []
        remaining = results.copy()
        
        while len(selected) < top_k and remaining:
            # Take the best remaining result
            best = remaining.pop(0)
            selected.append(best)
            
            # Apply diversity penalty to remaining results
            if "embedding" in best:
                best_emb = np.array(best["embedding"]).reshape(1, -1)
                
                for result in remaining:
                    if "embedding" in result:
                        result_emb = np.array(result["embedding"]).reshape(1, -1)
                        similarity = cosine_similarity(best_emb, result_emb)[0][0]
                        
                        # Penalize similar results
                        penalty = similarity * self.diversity_penalty
                        result["final_score"] = result.get("final_score", 0) * (1 - penalty)
                
                # Re-sort remaining
                remaining.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return selected
    
    def rerank_by_query(
        self,
        results: List[Dict],
        query_embedding: np.ndarray,
        chunk_embeddings: Dict[str, np.ndarray],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank results by direct query similarity.
        
        Useful as a second-pass ranking.
        """
        query_embedding = query_embedding.reshape(1, -1)
        
        for result in results:
            chunk_id = result.get("id")
            if chunk_id in chunk_embeddings:
                chunk_emb = np.array(chunk_embeddings[chunk_id]).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, chunk_emb)[0][0]
                result["rerank_score"] = float(similarity)
            else:
                result["rerank_score"] = result.get("final_score", 0)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:top_k]
    
    def filter_by_threshold(
        self,
        results: List[Dict],
        min_score: float = 0.1,
        score_key: str = "final_score"
    ) -> List[Dict]:
        """Filter results below a minimum score threshold."""
        return [r for r in results if r.get(score_key, 0) >= min_score]
    
    def format_results_for_context(
        self,
        results: List[Dict],
        include_metadata: bool = True,
        max_context_length: int = 4000
    ) -> str:
        """
        Format results as context string for LLM.
        
        Args:
            results: List of result dictionaries
            include_metadata: Whether to include source metadata
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            text = result.get("text", "")
            
            if include_metadata:
                source = result.get("source", "unknown")
                score = result.get("final_score", 0)
                chunk_id = result.get("id", f"chunk_{i}")
                
                part = f"[Source: {chunk_id}, Relevance: {score:.2f}]\n{text}\n"
            else:
                part = f"{text}\n"
            
            # Check length
            if total_length + len(part) > max_context_length:
                # Truncate if needed
                remaining = max_context_length - total_length
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break
            
            context_parts.append(part)
            total_length += len(part)
        
        return "\n---\n".join(context_parts)
    
    def get_result_citations(self, results: List[Dict]) -> List[Dict]:
        """
        Extract citation information from results.
        
        Returns list of citations with chunk IDs and scores.
        """
        citations = []
        
        for result in results:
            citation = {
                "chunk_id": result.get("id"),
                "score": result.get("final_score", 0),
                "source": result.get("source", "unknown"),
                "text_preview": result.get("text", "")[:100] + "..."
            }
            citations.append(citation)
        
        return citations
