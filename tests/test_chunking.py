"""
Tests for Semantic Chunking module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking.semantic_chunker import SemanticChunker
from chunking.buffer_merger import BufferMerger


class TestSemanticChunker:
    """Tests for SemanticChunker class."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance for testing."""
        return SemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            max_tokens=1024,
            sub_chunk_tokens=128,
            similarity_threshold=0.5
        )
    
    def test_initialization(self, chunker):
        """Test chunker initializes correctly."""
        assert chunker.max_tokens == 1024
        assert chunker.sub_chunk_tokens == 128
        assert chunker.similarity_threshold == 0.5
        assert chunker.model is not None
    
    def test_split_into_sentences(self, chunker):
        """Test sentence splitting."""
        text = "This is sentence one. This is sentence two. And this is three."
        sentences = chunker.split_into_sentences(text)
        
        assert len(sentences) >= 1
        assert all(len(s) > 10 for s in sentences)
    
    def test_estimate_tokens(self, chunker):
        """Test token estimation."""
        text = "This is a test sentence with several words."
        tokens = chunker.estimate_tokens(text)
        
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be less than chars
    
    def test_generate_embeddings(self, chunker):
        """Test embedding generation."""
        sentences = ["Hello world.", "This is a test."]
        embeddings = chunker.generate_embeddings(sentences)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Has embedding dimensions
    
    def test_semantic_grouping(self, chunker):
        """Test semantic grouping of sentences."""
        sentences = [
            "Dr. Ambedkar was a social reformer.",
            "He fought against caste discrimination.",
            "The weather is nice today.",
            "It might rain tomorrow."
        ]
        embeddings = chunker.generate_embeddings(sentences)
        groups = chunker.semantic_grouping(sentences, embeddings)
        
        assert len(groups) > 0
        assert all(isinstance(g, list) for g in groups)
        assert all(isinstance(i, int) for g in groups for i in g)


class TestBufferMerger:
    """Tests for BufferMerger class."""
    
    @pytest.fixture
    def merger(self):
        """Create a merger instance for testing."""
        return BufferMerger(buffer_size=2)
    
    def test_initialization(self, merger):
        """Test merger initializes correctly."""
        assert merger.buffer_size == 2
        assert merger.overlap_ratio == 0.2
    
    def test_sliding_window_chunks(self, merger):
        """Test sliding window chunking."""
        sentences = [f"Sentence {i}." for i in range(10)]
        chunks = merger.create_sliding_window_chunks(
            sentences, window_size=3, stride=2
        )
        
        assert len(chunks) > 0
        assert all("text" in c for c in chunks)
        assert all("sentences" in c for c in chunks)
    
    def test_add_cross_chunk_context(self, merger):
        """Test adding cross-chunk context."""
        chunks = [
            {"id": "1", "text": "First chunk", "sentences": ["First chunk"]},
            {"id": "2", "text": "Second chunk", "sentences": ["Second chunk"]},
            {"id": "3", "text": "Third chunk", "sentences": ["Third chunk"]}
        ]
        
        enhanced = merger.add_cross_chunk_context(chunks, context_sentences=1)
        
        assert len(enhanced) == 3
        assert "context_before" in enhanced[1]
        assert "context_after" in enhanced[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
