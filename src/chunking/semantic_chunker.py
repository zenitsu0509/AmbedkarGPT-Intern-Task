"""
Semantic Chunker - Implementation of SEMRAG Algorithm 1

This module implements semantic chunking via cosine similarity of sentence embeddings.
It groups sentences into semantically coherent chunks while respecting token limits.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from tqdm import tqdm


class SemanticChunker:
    """
    Implements SEMRAG Algorithm 1: Semantic Chunking via cosine similarity.
    
    Steps:
    1. Extract text from PDF
    2. Split into sentences
    3. Generate sentence embeddings
    4. Group sentences by semantic similarity (cosine similarity)
    5. Apply buffer merging for contextual continuity
    6. Enforce token limits (max 1024 tokens, sub-chunks ~128 tokens)
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_tokens: int = 1024,
        sub_chunk_tokens: int = 128,
        overlap_tokens: int = 32,
        similarity_threshold: float = 0.5,
        buffer_size: int = 3
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_model: Name of the sentence-transformers model
            max_tokens: Maximum tokens per chunk (default: 1024)
            sub_chunk_tokens: Target size for sub-chunks (default: 128)
            overlap_tokens: Overlap between sub-chunks (default: 32)
            similarity_threshold: Cosine similarity threshold for grouping (default: 0.5)
            buffer_size: Number of sentences for buffer context (default: 3)
        """
        self.embedding_model_name = embedding_model
        self.max_tokens = max_tokens
        self.sub_chunk_tokens = sub_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Approximate tokens per word (rough estimate)
        self.tokens_per_word = 1.3
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        print(f"Extracting text from: {pdf_path}")
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Reading PDF pages")):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        full_text = "\n".join(text_content)
        print(f"Extracted {len(full_text)} characters from {len(text_content)} pages")
        return full_text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex-based approach."""
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Split by sentence boundaries
        # Handle abbreviations and special cases
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out very short sentences and clean up
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Minimum sentence length
                cleaned_sentences.append(sent)
        
        print(f"Split into {len(cleaned_sentences)} sentences")
        return cleaned_sentences
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string."""
        words = len(text.split())
        return int(words * self.tokens_per_word)
    
    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences."""
        print("Generating sentence embeddings...")
        embeddings = self.model.encode(
            sentences,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarity between embeddings."""
        return cosine_similarity(embeddings)
    
    def semantic_grouping(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[List[int]]:
        """
        Group sentences based on semantic similarity (Algorithm 1 core logic).
        
        Uses cosine similarity to determine which sentences should be grouped together.
        """
        n = len(sentences)
        if n == 0:
            return []
        
        # Calculate similarity between consecutive sentences
        groups = []
        current_group = [0]
        
        for i in range(1, n):
            # Calculate similarity with previous sentence
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i-1].reshape(1, -1)
            )[0][0]
            
            # Also check similarity with group centroid
            group_embeddings = embeddings[current_group]
            group_centroid = np.mean(group_embeddings, axis=0).reshape(1, -1)
            centroid_sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                group_centroid
            )[0][0]
            
            # Check if current group would exceed token limit
            current_text = " ".join([sentences[j] for j in current_group])
            current_tokens = self.estimate_tokens(current_text)
            next_tokens = self.estimate_tokens(sentences[i])
            
            # Decision: add to current group or start new group
            if (sim >= self.similarity_threshold or centroid_sim >= self.similarity_threshold) \
                    and (current_tokens + next_tokens) <= self.max_tokens:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def apply_buffer_merging(
        self,
        sentences: List[str],
        groups: List[List[int]],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Apply buffer merging to preserve contextual continuity.
        
        Adds context from surrounding sentences to each chunk.
        """
        chunks = []
        
        for group_idx, group in enumerate(groups):
            # Get main sentences
            main_sentences = [sentences[i] for i in group]
            main_text = " ".join(main_sentences)
            
            # Get buffer sentences (context from before and after)
            buffer_before = []
            buffer_after = []
            
            # Buffer from previous group
            if group_idx > 0:
                prev_group = groups[group_idx - 1]
                buffer_indices = prev_group[-self.buffer_size:]
                buffer_before = [sentences[i] for i in buffer_indices]
            
            # Buffer from next group
            if group_idx < len(groups) - 1:
                next_group = groups[group_idx + 1]
                buffer_indices = next_group[:self.buffer_size]
                buffer_after = [sentences[i] for i in buffer_indices]
            
            # Calculate chunk embedding (mean of sentence embeddings)
            chunk_embedding = np.mean(embeddings[group], axis=0)
            
            chunk = {
                "id": f"chunk_{group_idx}",
                "text": main_text,
                "sentences": main_sentences,
                "sentence_indices": group,
                "buffer_before": buffer_before,
                "buffer_after": buffer_after,
                "full_context": " ".join(buffer_before + main_sentences + buffer_after),
                "token_count": self.estimate_tokens(main_text),
                "embedding": chunk_embedding.tolist()
            }
            chunks.append(chunk)
        
        return chunks
    
    def create_sub_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Create sub-chunks with overlap for chunks exceeding sub_chunk_tokens.
        
        Ensures smaller retrieval units while maintaining context.
        """
        sub_chunks = []
        sub_chunk_id = 0
        
        for chunk in chunks:
            text = chunk["text"]
            tokens = chunk["token_count"]
            
            if tokens <= self.sub_chunk_tokens:
                # Small enough, keep as is
                chunk["sub_chunk_id"] = f"sub_{sub_chunk_id}"
                chunk["parent_chunk_id"] = chunk["id"]
                sub_chunks.append(chunk)
                sub_chunk_id += 1
            else:
                # Split into sub-chunks with overlap
                words = text.split()
                words_per_sub = int(self.sub_chunk_tokens / self.tokens_per_word)
                overlap_words = int(self.overlap_tokens / self.tokens_per_word)
                
                start = 0
                while start < len(words):
                    end = min(start + words_per_sub, len(words))
                    sub_text = " ".join(words[start:end])
                    
                    # Generate embedding for sub-chunk
                    sub_embedding = self.model.encode([sub_text])[0]
                    
                    sub_chunk = {
                        "id": f"sub_{sub_chunk_id}",
                        "parent_chunk_id": chunk["id"],
                        "text": sub_text,
                        "buffer_before": chunk["buffer_before"] if start == 0 else [],
                        "buffer_after": chunk["buffer_after"] if end == len(words) else [],
                        "full_context": sub_text,
                        "token_count": self.estimate_tokens(sub_text),
                        "embedding": sub_embedding.tolist()
                    }
                    sub_chunks.append(sub_chunk)
                    sub_chunk_id += 1
                    
                    # Move start with overlap
                    start = end - overlap_words
                    if start >= end:
                        break
        
        return sub_chunks
    
    def process_document(self, pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a PDF document through the full semantic chunking pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (chunks, sub_chunks)
        """
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Split into sentences
        sentences = self.split_into_sentences(text)
        
        # Step 3: Generate embeddings
        embeddings = self.generate_embeddings(sentences)
        
        # Step 4: Semantic grouping (Algorithm 1 core)
        print("Performing semantic grouping...")
        groups = self.semantic_grouping(sentences, embeddings)
        print(f"Created {len(groups)} semantic groups")
        
        # Step 5: Apply buffer merging
        print("Applying buffer merging...")
        chunks = self.apply_buffer_merging(sentences, groups, embeddings)
        
        # Step 6: Create sub-chunks with overlap
        print("Creating sub-chunks...")
        sub_chunks = self.create_sub_chunks(chunks)
        print(f"Created {len(sub_chunks)} sub-chunks")
        
        return chunks, sub_chunks
    
    def save_chunks(
        self,
        chunks: List[Dict],
        output_path: str,
        include_embeddings: bool = False
    ):
        """Save chunks to a JSON file."""
        # Optionally remove embeddings to reduce file size
        if not include_embeddings:
            chunks_to_save = []
            for chunk in chunks:
                chunk_copy = chunk.copy()
                chunk_copy.pop("embedding", None)
                chunks_to_save.append(chunk_copy)
        else:
            chunks_to_save = chunks
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks_to_save)} chunks to {output_path}")


def main():
    """Example usage of the semantic chunker."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    chunking_config = config["chunking"]
    
    # Initialize chunker
    chunker = SemanticChunker(
        embedding_model=chunking_config["embedding_model"],
        max_tokens=chunking_config["max_tokens"],
        sub_chunk_tokens=chunking_config["sub_chunk_tokens"],
        overlap_tokens=chunking_config["overlap_tokens"],
        similarity_threshold=chunking_config["similarity_threshold"],
        buffer_size=chunking_config["buffer_size"]
    )
    
    # Process document
    pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "Ambedkar_book.pdf"
    chunks, sub_chunks = chunker.process_document(str(pdf_path))
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    chunker.save_chunks(chunks, output_dir / "chunks.json")
    chunker.save_chunks(sub_chunks, output_dir / "sub_chunks.json")
    
    print(f"\nChunking complete!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total sub-chunks: {len(sub_chunks)}")


if __name__ == "__main__":
    main()
