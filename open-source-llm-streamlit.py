import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate, StorageContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
import os
import sys
import logging

from llama_index.core import Settings 
from llama_index.core import load_index_from_storage




# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# Load documents
reader = SimpleDirectoryReader(input_files=["C:/Users/Naman/Desktop/mistral-rag/files/ny_collaborative_protocols_v23.1.pdf"])
documents = reader.load_data()



#Embedding model
Settings.embed_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large")

embed_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    

#ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model="local")



# Initialize LLM
llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    model_path='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    temperature=0,
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
Settings.llm=llm


query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is the protocol for smoke inhalation")

print("r")          
print(response)


response = query_engine.query("What is apgar score")

print(response)



