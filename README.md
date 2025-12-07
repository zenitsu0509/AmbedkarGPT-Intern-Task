# AmbedkarGPT - SEMRAG-based RAG System

A fully functional RAG (Retrieval-Augmented Generation) system for answering questions about Dr. B.R. Ambedkar's works, built following the **SEMRAG research paper** architecture.

## 🎯 Features

- **Semantic Chunking** (Algorithm 1): Groups sentences by cosine similarity with buffer merging
- **Knowledge Graph**: Entities and relationships with community detection (Louvain/Leiden)
- **Local RAG Search** (Equation 4): Entity-based chunk retrieval
- **Global RAG Search** (Equation 5): Community-based retrieval
- **LLM Integration**: Mistral 7B via Ollama for answer generation

## 📋 Requirements

- Python 3.9+
- Ollama with Mistral 7B model
- ~4GB RAM for embeddings and graph
- ~8GB VRAM for Mistral 7B (or CPU inference)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
cd ambedkargpt
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Ensure Ollama is Running

```bash
# Start Ollama server (in another terminal)
ollama serve

# Verify Mistral is available
ollama list

# Pull if not available
ollama pull mistral:7b
```

### 3. Ingest the Document

```bash
# Run ingestion pipeline
python run.py --ingest

# Or with custom PDF path
python run.py --ingest --pdf "../data/Ambedkar_book.pdf"
```

### 4. Ask Questions

```bash
# Interactive mode
python run.py --interactive

# Single query
python run.py --query "What were Ambedkar's views on caste?"
```

## 📁 Project Structure

```
ambedkargpt/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── run.py                   # Main entry point
├── data/
│   ├── Ambedkar_book.pdf    # Source document (copy here)
│   └── processed/           # Processed data (auto-generated)
│       ├── chunks.json
│       ├── sub_chunks.json
│       ├── knowledge_graph.pkl
│       ├── embeddings.pkl
│       └── communities.pkl
├── src/
│   ├── chunking/            # Semantic chunking (Algorithm 1)
│   │   ├── semantic_chunker.py
│   │   └── buffer_merger.py
│   ├── graph/               # Knowledge graph construction
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── retrieval/           # RAG search (Equations 4 & 5)
│   │   ├── local_search.py
│   │   ├── global_search.py
│   │   └── ranker.py
│   ├── llm/                 # LLM integration
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   └── pipeline/            # Main pipeline
│       └── ambedkargpt.py
└── tests/                   # Unit tests
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_integration.py
```

## ⚙️ Configuration

Edit `config.yaml` to adjust parameters:

```yaml
# Chunking parameters
chunking:
  embedding_model: "all-MiniLM-L6-v2"
  max_tokens: 1024
  similarity_threshold: 0.5

# Retrieval thresholds
retrieval:
  local:
    entity_similarity_threshold: 0.3  # τ_e
    chunk_similarity_threshold: 0.2   # τ_d
    top_k: 5

# LLM settings
llm:
  model: "mistral:7b"
  base_url: "http://localhost:11434"
  temperature: 0.7
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Usage Examples

### Python API

```python
from src.pipeline.ambedkargpt import AmbedkarGPT

# Initialize
gpt = AmbedkarGPT("config.yaml")

# Load processed data (if already ingested)
gpt.load_processed_data()

# Or ingest new document
gpt.ingest_document("path/to/pdf")

# Query
result = gpt.query("What is Ambedkar's view on education?")
print(result["answer"])
print(result["citations"])
```

### Command Line

```bash
# Ingest document
python run.py --ingest

# Query mode
python run.py --query "Explain Ambedkar's philosophy on equality"

# Interactive chat
python run.py --interactive

# Load and query (skip ingestion)
python run.py --load --query "What reforms did Ambedkar advocate?"
```

## 🎤 Live Demo Checklist

Before the interview demo:

1. ✅ **Environment Ready**
   - [ ] Virtual environment activated
   - [ ] All dependencies installed
   - [ ] spaCy model downloaded

2. ✅ **Ollama Running**
   - [ ] `ollama serve` running in background
   - [ ] `mistral:7b` model available

3. ✅ **Data Processed**
   - [ ] Run `python run.py --ingest` beforehand
   - [ ] Verify `data/processed/` folder has files

4. ✅ **Test Queries Ready**
   - "Who was Dr. B.R. Ambedkar?"
   - "What were Ambedkar's views on caste?"
   - "What role did Ambedkar play in the Indian Constitution?"
   - "What is the significance of education according to Ambedkar?"
   - "How did Ambedkar fight for social justice?"

5. ✅ **Quick Test**
   ```bash
   python run.py --load --query "Who was Ambedkar?"
   ```

## 🔧 Troubleshooting

### Ollama Connection Error
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list

# Pull model if missing
ollama pull mistral:7b
```

### Memory Issues
- Reduce `max_tokens` in config
- Use smaller embedding model
- Process document in batches

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Slow Ingestion
- First run downloads embedding model (~90MB)
- Graph construction may take 5-10 minutes for 94-page PDF
- Subsequent loads are fast (uses cached data)

## 📚 SEMRAG Implementation Details

### Algorithm 1: Semantic Chunking
- Sentence embeddings via `all-MiniLM-L6-v2`
- Cosine similarity grouping with threshold
- Buffer merging for context preservation
- Token limits enforced (max 1024, sub-chunks ~128)

### Equation 4: Local RAG Search
```
D_retrieved = Top_k({v ∈ V, g ∈ G | sim(v, Q+H) > τ_e ∧ sim(g, v) > τ_d})
```
- Entity similarity threshold τ_e = 0.3
- Chunk similarity threshold τ_d = 0.2

### Equation 5: Global RAG Search
```
D_retrieved = Top_k(⋃_{r ∈ R_Top-K(Q)} ⋃_{c_i ∈ C_r} (⋃_{p_j ∈ c_i} (p_j, score(p_j, Q))))
```
- Community detection via Louvain algorithm
- Top-K community summaries
- Chunk scoring within communities

## 📄 License

This project is for educational purposes as part of an internship assignment.

## 🙏 Acknowledgments

- SEMRAG Research Paper
- Dr. B.R. Ambedkar's Works
- Ollama Team
- Sentence Transformers
