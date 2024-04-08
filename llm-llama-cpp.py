import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage


from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
print(os.getcwd())

reader = SimpleDirectoryReader(
    input_files=[r"files\ny_collaborative_protocols_v23.1.pdf"]
)

documents = reader.load_data()

import chromadb
import torch
print(torch.cuda.is_available())

from llama_index.llms.llama_cpp import LlamaCPP



from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
    

available_gpus = torch.cuda.device_count()
print("gpu count",available_gpus)


def llm_sandbox(model_name,query):

        
    llm = LlamaCPP(
        #model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        #model_path='mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        model_path = model_name,
        temperature=0,
        max_new_tokens=256,
        context_window=4096,
        generate_kwargs={},
        # set n_gpu_layers to -1 to use all available GPUs
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    print(torch.cuda.is_available())

    #Settings.chunksize = 256
    Settings.llm=llm


    #use once index is saved
    #storage_context = StorageContext.from_defaults(persist_dir="embeddings")
    #index = load_index_from_storage(storage_context)

    #db = chromadb.PersistentClient(path="./chroma_db")


    #Embedding model
    Settings.embed_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large")

    embed_model=HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)    

    return response

import time
t1 = time.time()

response = llm_sandbox("mistral-7b-instruct-v0.1.Q4_K_M.gguf","Tell me about apgar score")
#print(response)
r1 = response
t2 = time.time()
print(t2 - t1)
t11= t2-t1


#t1 = time.time()
#response = llm_sandbox("llama-2-7b.Q3_K_L.gguf","Tell me about apgar score")
#print(response)
#t2 = time.time()
#print(t2 - t1)

t1 = time.time()
response = llm_sandbox("gemma-7b-it.Q4_0.gguf","Tell me about apgar score")
r3 = response
#print(response)
t2 = time.time()
print(t2 - t1)
t33 = t2-t1

t1 = time.time()
response = llm_sandbox("phi-2.Q4_K_M.gguf","Tell me about apgar score")
r4 = response
#print(response)
t2 = time.time()
print(t2 - t1)
t44 = t2-t1


print("mistral",r1,t11)
print("gemma",r3,t33)
print("phi",r4,t44)





    