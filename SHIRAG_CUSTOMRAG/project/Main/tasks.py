from project.Main.util import make_rest_call_userbackend,q_and_a_payload
from project.customrag.customragmain import create_vector_index
from project.customrag.llm import load_exsisting_database,get_answer
import os
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

EMBEDDING_MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'embedding_models.json')

# Load scenarios from JSON file
def load_embedding_model():
    if os.path.exists(EMBEDDING_MODEL_FILE_PATH):
        with open(EMBEDDING_MODEL_FILE_PATH, 'r') as f:
            return json.load(f)
    else:
        return {}



def create_customized_vb(document_path,vector_db_path,embedding_model,chunk_size,chunk_overlap,llm_model_path):
    try:
      embedding_model_path_dict = load_embedding_model()
      if embedding_model in embedding_model_path_dict .keys():
        print(f"Embedding_Model:{embedding_model}")
        embedding_model_path_actual = os.getenv(embedding_model)
        print(embedding_model_path_actual)
      create_vector_index(document_path,vector_db_path,embedding_model_path_actual,chunk_size,chunk_overlap,llm_model_path)
    except Exception as e:
        print(e)


def buildyourcustomrag(llm_model_path,embedding_model_path,vb_path,temp,search_threshold,top_docs,context_window,repetition_penality,top_k,top_p,user_question,question_id):
    try:
      embedding_model_path_dict = load_embedding_model()
      if embedding_model_path in embedding_model_path_dict .keys():
        print(f"Embedding_Model:{embedding_model_path}")
        embedding_model_path_actual = os.getenv(embedding_model_path)
        print(embedding_model_path_actual)
      load_exsisting_database(embedding_model_path_actual,vb_path,search_threshold,top_docs)
      #llm_response =chatbot(llm_model_path,max_tokens,temp,repetition_penality,top_k,top_p,user_question)
      #print(llm_response)
      rag_answer, without_raganswer = get_answer(llm_model_path,embedding_model_path,vb_path,temp,search_threshold,top_docs,context_window,repetition_penality,top_k,top_p,user_question)
      make_rest_call_userbackend(question_id,q_and_a_payload(rag_answer=rag_answer,without_rag=without_raganswer))
    except Exception as e:
       print(e)


