"""Command line interface for the RAG chatbot."""

from dotenv import load_dotenv

from .ingest import ingest_documents
from .chatbot import answer_question

def main() -> None:
    """Run an interactive chat session."""
    load_dotenv()
    vector_store = ingest_documents("data")
    print("RAG Chatbot. Type 'exit' to quit.")
    while True:
        try:
            query = input("You: ")
        except EOFError:
            break
        if query.strip().lower() in {"exit", "quit"}:
            break
        response = answer_question(vector_store, query)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()

