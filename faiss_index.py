import faiss
import numpy as np
import json
from time import time
import asyncio 
from datasets import Dataset
from typing import List
from dotenv import load_dotenv
import os
import pickle
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from langchain_core.documents import Document

# Import from the parent directory
import sys
sys.path.append('..')
from RAG_strategy_Taide import taide_llm, system_prompt, multi_query
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from add_vectordb import GetVectorStore

# Import RAGAS metrics
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision

# Load environment variables
load_dotenv('../environment.env')
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
document_table = "documents"

# Initialize Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embeddings and chat model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def download_embeddings():
    response = supabase.table(document_table).select("id, embedding, metadata, content").execute()
    embeddings = []
    ids = []
    metadatas = []
    contents = []
    for item in response.data:
        embedding = json.loads(item['embedding'])
        embeddings.append(embedding)
        ids.append(item['id'])
        metadatas.append(item['metadata'])
        contents.append(item['content'])
    return np.array(embeddings, dtype=np.float32), ids, metadatas, contents

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize embeddings for cosine similarity
    index.add(embeddings)
    return index

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    if os.path.exists(file_path):
        index = faiss.read_index(file_path)
        print(f"FAISS index loaded from {file_path}")
        return index
    return None

def save_metadata(ids, metadatas, contents, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((ids, metadatas, contents), f)
    print(f"Metadata saved to {file_path}")

def load_metadata(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            ids, metadatas, contents = pickle.load(f)
        print(f"Metadata loaded from {file_path}")
        return ids, metadatas, contents
    return None, None, None

def search_faiss(index, query_vector, k=4):
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]

class FAISSRetriever:
    def __init__(self, index, ids, metadatas, contents, embeddings_model):
        self.index = index
        self.ids = ids
        self.metadatas = metadatas
        self.contents = contents
        self.embeddings_model = embeddings_model

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embeddings_model.embed_query(query)
        _, indices = search_faiss(self.index, query_vector)
        return [
            Document(page_content=self.contents[i], metadata=self.metadatas[i])
            for i in indices
        ]

def load_qa_pairs():
    df = pd.read_csv("../QA_database_rows.csv")
    return df['Question'].tolist(), df['Answer'].tolist()

def faiss_query(question: str, retriever: FAISSRetriever) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
        system_prompt + "\n\n" +
        "Answer the following question based on this context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer in the same language as the question. If you don't know the answer, "
        "say 'I'm sorry, I don't have enough information to answer that question.'"
    )
    
    chain = prompt | taide_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

async def run_evaluation():
    faiss_index_path = "faiss_index.bin"
    metadata_path = "faiss_metadata.pkl"

    index = load_faiss_index(faiss_index_path)
    ids, metadatas, contents = load_metadata(metadata_path)

    if index is None or ids is None:
        print("FAISS index or metadata not found. Creating new index...")
        print("Downloading embeddings from Supabase...")
        embeddings_array, ids, metadatas, contents = download_embeddings()

        print("Creating FAISS index...")
        index = create_faiss_index(embeddings_array)

        save_faiss_index(index, faiss_index_path)
        save_metadata(ids, metadatas, contents, metadata_path)
    else:
        print("Using existing FAISS index and metadata.")

    print("Creating FAISS retriever...")
    faiss_retriever = FAISSRetriever(index, ids, metadatas, contents, embeddings)

    print("Creating original vector store...")
    original_vector_store = GetVectorStore(embeddings, supabase, document_table)
    original_retriever = original_vector_store.as_retriever(search_kwargs={"k": 4})

    questions, ground_truths = load_qa_pairs()

    for question, ground_truth in zip(questions, ground_truths):
        print(f"\nQuestion: {question}")

        start_time = time()
        faiss_answer = faiss_query(question, faiss_retriever)
        faiss_docs = faiss_retriever.get_relevant_documents(question)
        faiss_time = time() - start_time
        print(f"FAISS Answer: {faiss_answer}")
        print(f"FAISS Time: {faiss_time:.4f} seconds")

        start_time = time()
        original_answer, original_docs = multi_query(question, original_retriever, chat_history=[])
        original_time = time() - start_time
        print(f"Original Answer: {original_answer}")
        print(f"Original Time: {original_time:.4f} seconds")

        faiss_datasets = {
            "question": [question],
            "answer": [faiss_answer],
            "contexts": [[doc.page_content for doc in faiss_docs]],
            "ground_truth": [ground_truth]
        }
        faiss_evalsets = Dataset.from_dict(faiss_datasets)

        faiss_result = evaluate(
            faiss_evalsets,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )

        print("FAISS RAGAS Evaluation:")
        print(faiss_result.to_pandas())

        original_datasets = {
            "question": [question],
            "answer": [original_answer],
            "contexts": [[doc.page_content for doc in original_docs]],
            "ground_truth": [ground_truth]
        }
        original_evalsets = Dataset.from_dict(original_datasets)

        original_result = evaluate(
            original_evalsets,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )

        print("Original RAGAS Evaluation:")
        print(original_result.to_pandas())

    print("\nPerformance comparison complete.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())