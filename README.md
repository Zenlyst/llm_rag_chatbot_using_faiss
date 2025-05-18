# FAISS Indexing and Retrieval System

The faiss_index.py script creates a document retrieval system that compares FAISS indexing with OpenAI embeddings in a RAG framework. It uses Supabase for data storage, applies RAGAS metrics to measure which method works better. The project uses the Taide LLM model, which is designed for Taiwanese content and provides responses in Traditional Chinese. 

---

## Features

- **Taiwanese-culture-tuned chatbot**
    - Using [Taide LLM model](https://en.taide.tw/) to create a chat-bot that focuses on traditional Chinese and Taiwanese-cultural-localised content.

- **FAISS-Based Retrieval**:
    - Uses FAISS for fast, cosine-similarity-based vector search.
    - Supports creating, saving, and loading FAISS indices.

- **Using RAG**
    - Implements Retrieval-Augmented Generation (RAG) to combine document retrieval with language model generation.
    - Uses the multi_query function to generate multiple variations of a user query, improving the quality of retrieved documents.
    - Leverages the retrieved documents as context for generating accurate and context-aware answers using the Taide LLM model.
    - Ensures that the assistant provides responses in the user's language (Traditional Chinese or English) based on the input query.

- **Integration with Supabase**:
    - Downloads embeddings, metadata, and document content from a Supabase database.

- **Evaluation Framework**:
    - Compares FAISS index with OpenAI embeddings
    - Evaluates performance using RAGAS metrics:
        - context_precision
        - faithfulness
        - answer_relevancy
        - context_recall

- **Asynchronous Execution**:
    - Uses Python's asyncio for efficient evaluation of multiple queries.

---

## Requirements

**Python Libraries**

Install the required dependencies using `pip`:

```pip install faiss-cpu numpy pandas langchain supabase datasets ragas python-dotenv```

**Environment Variables**

Create an `.env` file in the parent directory with the following variables:

```
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
OPENAI_API_KEY=<your_openai_api_key>
```

**Prerequisites for Using Taide LLM**

- **Ollama Installation**:

    - Ensure that the ollama CLI is installed and configured on your system.

- **Model Availability**:

    - The `taide` model must be downloaded and available for use with the ollama CLI.

--- 

## How It Works

1. **Downloading Embeddings**
The script fetches embeddings, metadata, and document content from a Supabase table:
`embeddings_array, ids, metadatas, contents = download_embeddings()`

2. **Creating and Saving a FAISS Index**
The FAISS index is created using the downloaded embeddings:

``` 
index = create_faiss_index(embeddings_array)
save_faiss_index(index, "faiss_index.bin")
```

3. **Loading an Existing FAISS Index**
If a saved FAISS index exists, it is loaded:
`index = load_faiss_index("faiss_index.bin")`

4. **Retrieving Documents**
The FAISSRetriever class retrieves relevant documents for a given query:
```
retriever = FAISSRetriever(index, ids, metadatas, contents, embeddings)
docs = retriever.get_relevant_documents(query)
```

5. **Generating Multiple Query Variations**
The multi_query function from RAG_strategy_Taide.py generates multiple variations of a user query to improve retrieval performance:
`final_answer, reference_docs = multi_query(question, retriever)`

6. **Evaluating Performance**
The script evaluates the FAISS-based retriever and compares it with an original vector store using RAGAS metrics:
```faiss_result = evaluate(faiss_evalsets, metrics=[context_precision, faithfulness, answer_relevancy, context_recall])```

--- 

## Key Components from RAG_strategy_Taide.py

1. `system_prompt`
Defines the behavior of the AI assistant:
system_prompt = "你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"

2. `taide_llm`
A custom language model (OllamaChatModel) that processes queries and generates responses:
`taide_llm = OllamaChatModel(model_name="taide-local")`

3. `multi_query`
Generates multiple variations of a user query to improve document retrieval:
`final_answer, reference_docs = multi_query(question, retriever)`

---

## Usage

Running the Script

To execute the script and evaluate the retrieval system:

`python faiss_index.py`

---

**Example Output**

The script compares the FAISS-based retriever with the original vector store and outputs evaluation metrics for each query:

```
Question: What is the purpose of the CEV system?
FAISS Answer: The CEV system supports inventory tracking and compliance.
FAISS Time: 0.1234 seconds
Original Answer: The CEV system ensures compliance with ISO standards.
Original Time: 0.5678 seconds

FAISS RAGAS Evaluation:
   context_precision  faithfulness  answer_relevancy  context_recall
0              0.85          0.90             0.88           0.80

Original RAGAS Evaluation:
   context_precision  faithfulness  answer_relevancy  context_recall
0              0.80          0.85             0.83           0.75
```

--- 

## Notes
    - Ensure that the Supabase table contains the required fields: id, embedding, metadata, and content.
    - The FAISS index uses cosine similarity for document retrieval.
    - Modify the system_prompt in the script to customize the query-answering behavior.