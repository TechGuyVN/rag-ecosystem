# RAG Chatbot Example

This example demonstrates a minimal Retrieval-Augmented Generation (RAG) chatbot
using OpenAI models and a Qdrant vector store.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the OpenAI API key in a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your_key" > .env
   ```

## Usage

Index the sample documents and start chatting:

```bash
python -m rag_chatbot.main
```

Type a question related to the indexed documents and the bot will respond using
retrieved context from Qdrant.

