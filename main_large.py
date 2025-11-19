from pathlib import Path
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.document_loaders import DirectoryLoader, TextLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import Ollama
    from langchain.vectorstores import Chroma

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_DIR = Path("documents")
PERSIST_DIR = Path("chroma_store_large")
PROMPT_TEMPLATE = """You are a knowledgeable assistant answering questions about Dr. B. R. Ambedkar's works.
Use only the provided context from his writings. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def _format_docs(documents):
    """Concatenate retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def index_documents():
    """Build and persist the vector store from all documents in the documents/ folder."""
    if not DOCUMENTS_DIR.exists():
        print(f"Error: {DOCUMENTS_DIR} folder not found.")
        print("Create a 'documents' folder and add your text files there.")
        sys.exit(1)

    text_files = list(DOCUMENTS_DIR.glob("**/*.txt"))
    if not text_files:
        print(f"Error: No .txt files found in {DOCUMENTS_DIR}")
        sys.exit(1)

    print(f"Found {len(text_files)} text file(s). Starting indexing...")

    # Load all documents
    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and vector store
    print("Creating embeddings (this may take several minutes for large datasets)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Process in batches to avoid memory issues
    batch_size = 1000
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(PERSIST_DIR),
            )
        else:
            vectorstore.add_documents(batch)

    print(f"Indexing complete! Vector store saved to {PERSIST_DIR}")


def load_qa_chain():
    """Load the existing vector store and create a QA chain."""
    if not PERSIST_DIR.exists():
        print(f"Error: Vector store not found at {PERSIST_DIR}")
        print("Run with --index flag first to build the index.")
        sys.exit(1)

    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    llm = Ollama(model="mistral")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10},  
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    """Main entry point - index or query based on command-line arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "--index":
        index_documents()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--reindex":
        if PERSIST_DIR.exists():
            import shutil
            shutil.rmtree(PERSIST_DIR)
            print(f"Removed existing index at {PERSIST_DIR}")
        index_documents()
        return

    # Query mode
    qa_chain = load_qa_chain()
    print("Ambedkar Works Q&A (Large Collection) — type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        print("Searching...")
        response = qa_chain.invoke(question)
        print(f"Answer: {response.strip()}\n")


if __name__ == "__main__":
    main()
