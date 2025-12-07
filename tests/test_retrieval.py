"""
Tests for Retrieval modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.local_search import LocalGraphRAGSearch
from retrieval.global_search import GlobalGraphRAGSearch
from retrieval.ranker import ResultRanker


class TestLocalGraphRAGSearch:
    """Tests for Local RAG Search (Equation 4)."""
    
    @pytest.fixture
    def local_search(self):
        """Create local search instance."""
        return LocalGraphRAGSearch(
            entity_similarity_threshold=0.3,
            chunk_similarity_threshold=0.2,
            top_k=5
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Sample embeddings (384-dim like MiniLM)
        dim = 384
        
        entity_embeddings = {
            "ambedkar": np.random.randn(dim),
            "caste": np.random.randn(dim),
            "constitution": np.random.randn(dim)
        }
        
        chunk_embeddings = {
            "chunk_1": np.random.randn(dim),
            "chunk_2": np.random.randn(dim),
            "chunk_3": np.random.randn(dim)
        }
        
        entity_to_chunks = {
            "ambedkar": ["chunk_1", "chunk_2"],
            "caste": ["chunk_1", "chunk_3"],
            "constitution": ["chunk_2", "chunk_3"]
        }
        
        chunks = [
            {"id": "chunk_1", "text": "About Ambedkar and caste system."},
            {"id": "chunk_2", "text": "Ambedkar and the constitution."},
            {"id": "chunk_3", "text": "Caste and constitutional reforms."}
        ]
        
        return {
            "entity_embeddings": entity_embeddings,
            "chunk_embeddings": chunk_embeddings,
            "entity_to_chunks": entity_to_chunks,
            "chunks": chunks
        }
    
    def test_initialization(self, local_search):
        """Test local search initializes correctly."""
        assert local_search.entity_threshold == 0.3
        assert local_search.chunk_threshold == 0.2
        assert local_search.top_k == 5
    
    def test_find_relevant_entities(self, local_search, sample_data):
        """Test entity relevance finding."""
        query_embedding = np.random.randn(384)
        
        # Make query similar to 'ambedkar' entity
        query_embedding = sample_data["entity_embeddings"]["ambedkar"] + np.random.randn(384) * 0.1
        
        results = local_search._find_relevant_entities(
            query_embedding.reshape(1, -1),
            sample_data["entity_embeddings"]
        )
        
        assert isinstance(results, list)
        # Should find the similar entity
        if results:
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_search(self, local_search, sample_data):
        """Test full local search."""
        query_embedding = np.random.randn(384)
        
        results = local_search.search(
            query_embedding=query_embedding,
            entity_embeddings=sample_data["entity_embeddings"],
            chunk_embeddings=sample_data["chunk_embeddings"],
            entity_to_chunks=sample_data["entity_to_chunks"],
            chunks=sample_data["chunks"],
            top_k=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3


class TestGlobalGraphRAGSearch:
    """Tests for Global RAG Search (Equation 5)."""
    
    @pytest.fixture
    def global_search(self):
        """Create global search instance."""
        return GlobalGraphRAGSearch(
            top_k_communities=3,
            top_k_chunks=5
        )
    
    @pytest.fixture
    def sample_communities(self):
        """Create sample community data."""
        dim = 384
        
        community_info = [
            {
                "community_id": 0,
                "size": 5,
                "summary": "Social reform and caste issues",
                "chunk_ids": ["chunk_1", "chunk_2"]
            },
            {
                "community_id": 1,
                "size": 3,
                "summary": "Constitutional matters",
                "chunk_ids": ["chunk_3"]
            }
        ]
        
        community_embeddings = {
            0: np.random.randn(dim),
            1: np.random.randn(dim)
        }
        
        chunk_embeddings = {
            "chunk_1": np.random.randn(dim),
            "chunk_2": np.random.randn(dim),
            "chunk_3": np.random.randn(dim)
        }
        
        chunks = [
            {"id": "chunk_1", "text": "Social reform text."},
            {"id": "chunk_2", "text": "Caste issues text."},
            {"id": "chunk_3", "text": "Constitution text."}
        ]
        
        return {
            "community_info": community_info,
            "community_embeddings": community_embeddings,
            "chunk_embeddings": chunk_embeddings,
            "chunks": chunks
        }
    
    def test_initialization(self, global_search):
        """Test global search initializes correctly."""
        assert global_search.top_k_communities == 3
        assert global_search.top_k_chunks == 5
    
    def test_search(self, global_search, sample_communities):
        """Test full global search."""
        query_embedding = np.random.randn(384)
        
        communities, chunks = global_search.search(
            query_embedding=query_embedding,
            community_info=sample_communities["community_info"],
            community_embeddings=sample_communities["community_embeddings"],
            chunk_embeddings=sample_communities["chunk_embeddings"],
            chunks=sample_communities["chunks"]
        )
        
        assert isinstance(communities, list)
        assert isinstance(chunks, list)


class TestResultRanker:
    """Tests for Result Ranker."""
    
    @pytest.fixture
    def ranker(self):
        """Create ranker instance."""
        return ResultRanker(local_weight=0.6, global_weight=0.4)
    
    def test_initialization(self, ranker):
        """Test ranker initializes correctly."""
        assert ranker.local_weight == 0.6
        assert ranker.global_weight == 0.4
    
    def test_combine_results(self, ranker):
        """Test combining local and global results."""
        local_results = [
            {"id": "chunk_1", "combined_score": 0.8, "text": "Local 1"},
            {"id": "chunk_2", "combined_score": 0.6, "text": "Local 2"}
        ]
        
        global_results = [
            {"id": "chunk_2", "global_score": 0.7, "text": "Global 2"},
            {"id": "chunk_3", "global_score": 0.5, "text": "Global 3"}
        ]
        
        combined = ranker.combine_results(local_results, global_results, top_k=3)
        
        assert isinstance(combined, list)
        assert len(combined) <= 3
        assert all("final_score" in r for r in combined)
    
    def test_format_results_for_context(self, ranker):
        """Test formatting results for LLM context."""
        results = [
            {"id": "chunk_1", "final_score": 0.8, "text": "Some text here."},
            {"id": "chunk_2", "final_score": 0.6, "text": "More text here."}
        ]
        
        context = ranker.format_results_for_context(results)
        
        assert isinstance(context, str)
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
