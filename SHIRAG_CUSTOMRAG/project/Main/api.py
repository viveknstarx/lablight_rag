import json
import os
from datetime import datetime
from time import sleep
from pydantic import BaseModel
from typing import Optional
import pytz
from fastapi import FastAPI, BackgroundTasks,UploadFile, HTTPException, File, Form
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from project.Main.tasks import create_customized_vb,buildyourcustomrag
from fastapi.middleware.cors import CORSMiddleware
import random 
import string


app = FastAPI(title='SHI CUSTOMIZED RAG')

# TODO: Add explict list of allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class customized_rag_vectordb_creation(BaseModel):
    documents_path :str 
    embedding_model_path:str
    vectorindex_name:str
    chunk_size:int
    chunk_overlap:int
    llm_model_path :str

    # {"documents_path": "uploaded_files/66866e9559aec1c5d6021a26/66866ebf59aec1c5d6021a29/string", "vectorindex_name": "string__string__string__0__0", "embedding_model_path": "string", "llm_model_path": "string", "chunk_size": 0, "chunk_overlap": 0}

class customized_llm_creation(BaseModel):
    llm_model_path:str
    embedding_model_path:str
    vb_path:str
    temp:float=0.7
    search_threshold:float=0.3
    top_docs:int=3
    max_tokens:int=500
    repetition_penality:float=1.5
    top_k:int=40
    top_p:float=0.9
    user_question:str
    question_id:str

@app.post("/customized_rag/create_vector_db")
async def create_vectord_db(request:customized_rag_vectordb_creation,backgroundtasks:BackgroundTasks):
  res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))

  vector_db_path = request.documents_path+"/"+res
  print(vector_db_path)
  backgroundtasks.add_task(create_customized_vb,request.documents_path,vector_db_path,request.embedding_model_path,request.chunk_size,request.chunk_overlap,request.llm_model_path)
  sleep(0.5)
  return{'message':'succesfully started creating vectordb','vectordbpath':vector_db_path}
#with msge ,vectordbpath

@app.post("/customized_rag/ask_question")
async def ask_question(request:customized_llm_creation,backgroundtasks:BackgroundTasks):
    backgroundtasks.add_task(buildyourcustomrag,request.llm_model_path,request.embedding_model_path,request.vb_path,request.temp,request.search_threshold,request.top_docs,request.max_tokens,request.repetition_penality,request.top_k,request.top_p,request.user_question,request.question_id)
    return {'message':'started creating your custom rag'}

