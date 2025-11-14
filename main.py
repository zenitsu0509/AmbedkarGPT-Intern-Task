"""Simple command-line RAG app for Ambedkar speech."""
from pathlib import Path
import shutil

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

try:  # LangChain split most integrations into langchain-community>=0.3
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.vectorstores import Chroma
except ImportError:  # pragma: no cover - fallback for older LangChain installs
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import Ollama
    from langchain.vectorstores import Chroma

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for <0.3
    from langchain.text_splitters import CharacterTextSplitter

SPEECH_PATH = Path("speech.txt")
PERSIST_DIR = Path("chroma_store")
PROMPT_TEMPLATE = """You are a concise assistant that answers questions about Dr. B. R. Ambedkar's speech.\nUse only the provided context. If the answer is not contained in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""


def _format_docs(documents):
    """Concatenate retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def build_qa_chain():
    """Construct a lightweight RAG chain over the local speech transcript."""
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    loader = TextLoader(str(SPEECH_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    llm = Ollama(model="mistral")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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


def main() -> None:
    if not SPEECH_PATH.exists():
        raise FileNotFoundError("speech.txt is required in the project root")

    qa_chain = build_qa_chain()
    print("Ambedkar Q&A — type 'exit' to stop.")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        response = qa_chain.invoke(question)
        print(f"Answer: {response.strip()}\n")


if __name__ == "__main__":
    main()
