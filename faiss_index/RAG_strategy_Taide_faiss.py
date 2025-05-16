from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI 
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.globals import set_llm_cache
from langchain import PromptTemplate
import subprocess
import json
from typing import Any, List, Optional, Dict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
import os
from dotenv import load_dotenv
load_dotenv('environment.env')

from langchain.cache import SQLiteCache
from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_llm_cache

import requests
import openai
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
URI = os.getenv("SUPABASE_URI") 

# 設置緩存，以減少對API的重複請求。使用SQLite 
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

system_prompt: str = "您是一位來自台灣的 AI 助理，名字是 TAIDE，致力於以台灣人的視角協助使用者，並以繁體中文回答所有問題。"

class OllamaChatModel(BaseChatModel):
    model_name: str = Field(default="taide-local")

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                 formatted_messages.append({"role": "system", "content": msg.content})

        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        for msg in formatted_messages:
            if msg['role'] == 'user':
                prompt += f"{msg['content']} [/INST]"
            elif msg['role'] == "assistant":
                prompt += f"{msg['content']} </s><s>[INST]"

        command = ["ollama", "run", self.model_name, prompt]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Ollama command failed: {result.stderr}")
        
        content = result.stdout.strip()

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "ollama-chat-model"
    
taide_llm = OllamaChatModel(model_name="taide-local")

def multi_query(question, retriever, chat_history):
    """
    Generates multiple query variations for a given question, retrieves documents, 
    and generates a final answer using a RAG prompt.

    Args:
        question (str): The original user question.
        retriever: The retriever object for document retrieval.
        chat_history (list): The conversation history.

    Returns:
        tuple: A tuple containing the final answer and the retrieved documents.
    """
    def multi_query_chain(num_varitaions=3):
        template = f"""You are an AI language model assistant. Your task is to generate {num_varitaions} 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. 

        You must return original question also, which means that you return 1 original version + {num_varitaions} different versions = {num_varitaions + 1} questions.
        
        Original question: {{question}}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives 
            | taide_llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

        return generate_queries

    def get_unique_union(documents: List[list]):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    _search_query = get_search_query()
    modified_question = _search_query.invoke({"question":question, "chat_history": chat_history})
    # print(modified_question)

    generate_queries = multi_query_chain()

    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":modified_question})

    answer = multi_query_rag_prompt(retrieval_chain, modified_question)

    return answer, docs

def multi_query_rag_prompt(retrieval_chain, question):
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    Output in user's language. If the question is in zh-tw, then the output will be in zh-tw. If the question is in English, then the output will be in English.
    You should not mention anything about "根據提供的文件內容" or other similar terms.
    If you don't know the answer, just say that "很抱歉，目前我無法回答您的問題，請將您的詢問發送至 help@test-email.com 以便獲得更進一步的幫助，謝謝。I'm sorry I cannot answer your question. Please send your question to help@test-email.com for further assistance. Thank you."
    """

    prompt = ChatPromptTemplate.from_template(template)
    context = retrieval_chain.invoke({"question": question})
    # print(f"Retrieved context: {context[:200]}...")  # Print first 200 chars of context

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | taide_llm
        | StrOutputParser()
    )

    print(f"Sending question to model: {question}")
    try:
        answer = final_rag_chain.invoke({"question": question})
        # print(f"Received answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error invoking RAG chain for question '{question}': {e}")
        return "Error occurred while processing the question."

def get_search_query():
    _template = """Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    - The rewritten query should be in its original language.
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {chat_history}
    
    Original query: [{question}]
    
    Rewritten query: 
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI()
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x : x["question"]),
    )

    return _search_query