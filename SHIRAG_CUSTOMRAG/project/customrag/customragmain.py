from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from project.customrag.llm import model_loading
import string
import random

#laods the embedding model 
def intialize_emb(model_path):
    embedding=HuggingFaceEmbeddings(model_name = model_path)
    #returning model intiation for embedding creation
    return embedding


def create_vector_index(documents_path,vector_db_path,embedding_model_path,chunksize,chunkoverlap,llm_model_path):
    print(documents_path)
    loader = DirectoryLoader(documents_path, glob="./*.txt", loader_cls=TextLoader)
    documents1 = loader.load()
    print(documents1)

    loader = DirectoryLoader(documents_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents2 = loader.load()
    print(documents2)

    whole_docs = documents1+documents2
    print("whole documents",whole_docs)

    #splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunksize, chunk_overlap=chunkoverlap)
    texts = text_splitter.split_documents(whole_docs)
    #getting folderpath and intitaing the persistant directory to save the embedding database
    
    
    embedding = intialize_emb(embedding_model_path)
    global vectordb
    vectordb= Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=vector_db_path)
    
    vectordb.persist()
    #global retriever 
    #retriever = vectordb.as_retriever()
    # global inference_model
    # inference_model = model_loading(llm_model_path)
    # with open('inference_model.txt', 'w') as file:
    #     # Write the provided text to the file
    #     file.write(f'{inference_model}')
    #print(f"{inference_model}")
    print('VectorDB created succesfully')
