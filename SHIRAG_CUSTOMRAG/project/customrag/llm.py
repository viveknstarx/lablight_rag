from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gc
import torch
import os


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# def intialize_emb(model_name_1):
#     embedding=HuggingFaceEmbeddings(model_name = model_name_1)
#     #returning model intiation for embedding creation
#     return embedding


# def chatbot(max_tokens,temp,repetition_penality,top_k,top_p,input_text):
#     print(model,tokenizer)
#     global llm
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=max_tokens,temperature=temp,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penality)
#     print(pipe)
#     llm = HuggingFacePipeline(pipeline=pipe)
#     response = llm.invoke(input_text)
#     print('withoutrag',response)
#     return response
# def model_loading(llm_model_path):
#     global model_name_1
#     model_name_1 = llm_model_path
#     return model_name_1

# inference_model =model_loading(llm_model_path)

# def chatbot(llm_model_path,max_tokens,temp,repetition_penality,top_k,top_p,input_text):
#     global llm
#     # with open('inference_model.txt', 'r') as file:
#     #     # Read the content of the file
#     #     model_name = file.read().strip()
#     #     print(model_name)
#     llm = Ollama(model= llm_model_path,temperature=temp,num_ctx=max_tokens,num_predict = 128,repeat_penalty = repetition_penality,top_k =top_k,top_p =top_p,)
#     #llm = Ollama(model= llm_model_path,temperature=temp,)
#     # llm = Ollama(
#     #      model= "llama3",
#     #      temperature=0.9 ,
#     #      num_predict = 4096,
#     #      repeat_penalty = 1.5,
#     #      top_k =50,
#     #      top_p =1,)
#     # #model= model_name
#     # model = "llama3",
#     # temperature=temp
#     # num_predict = max_tokens
#     # repeat_penalty = repetition_penality
#     # top_k =top_k
#     # top_p =top_p
#     print(llm_model_path, temp, max_tokens, repetition_penality, top_k, top_p)  # assuming you have Ollama installed and have llama3 model pulled with ollama pull llama3
#     response=llm.invoke(input_text)
#     print(input_text)
#     print (response)
#     return response

class Chatbot:

    def __init__(self, model, temperature, num_ctx, num_predict, repeat_penalty, top_k, top_p):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        llm = Ollama(
            model=self.model,
            temperature=self.temperature,
            num_predict=self.num_predict,
            repeat_penalty=self.repeat_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        return llm

    def stop_instance(self):
        # Add any necessary cleanup code here
        self.llm = None

    def update_parameters(self, model=None, temperature=None, num_ctx=None, num_predict=128, repeat_penalty=None, top_k=None, top_p=None):

        # Stop the current instance
        self.stop_instance()

        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if num_ctx is not None:
            self.num_ctx = num_ctx
        if num_predict is not None:
            self.num_predict = num_predict
        if repeat_penalty is not None:
            self.repeat_penalty = repeat_penalty
        if top_k is not None:
            self.top_k = top_k
        if top_p is not None:
            self.top_p = top_p
        # Reinitialize llm with updated parameters
        self.llm = self.initialize_llm()

    def invoke(self, input_text):
        response = self.llm.invoke(input_text)
        return response


# class EmbeddingsInitializer:

#     def __init__(self,model_name):
#         self.model_name = model_name
#         self.embedding = self.initialize_emb()

#     def initialize_emb(self):
#         embedding = HuggingFaceEmbeddings(model_name=self.model_name)
#         return embedding

#     def stop_instance(self):
#         # Add any necessary cleanup code here
#         self.embedding = None


class EmbeddingsInitializer:

    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding = self.initialize_emb()

    def initialize_emb(self):
        embedding = HuggingFaceEmbeddings(model_name=self.model_name)
        return embedding

    def stop_instance(self):
        # Clean up the embedding model to free up memory
        if self.embedding is not None:
            # # Clear attributes of the embedding model, if any
            # for attr in dir(self.embedding):
            #     if not attr.startswith('__'):
            #         setattr(self.embedding, attr, None)
            # del self.embedding
            # # Set the embedding to None
            # self.embedding = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Clear Python's reference to any other objects
            for obj in gc.get_objects():
                try:
                    if isinstance(obj, torch.Tensor):
                        del obj
                except:
                    pass

            # Another garbage collection to ensure all objects are cleaned up
            gc.collect()
            self.embedding = None


# def shutdown_emb(embedding):
#     # Delete the embedding model to free up memory
#     del embedding
#     # Clear any remaining references and run garbage collection
#     gc.collect()
#     # If CUDA is being used, clear the CUDA cache
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


    def update_parameters(self, model_name=None):

        # Stop the current instance
        self.stop_instance()

        if model_name is not None:
            self.model_name = model_name
        # Reinitialize embedding with updated parameters
        self.embedding = self.initialize_emb()


global CHATBOT_INSTANCE
CHATBOT_INSTANCE = Chatbot(model="llama3.1", temperature=0.9, num_ctx=4096,
                           num_predict=128, repeat_penalty=1.5, top_k=40, top_p=1)


global EMBEDDING_INSTANCE
EMBEDDING_INSTANCE = EmbeddingsInitializer(
    model_name=os.getenv('bgelarge'))


def load_exsisting_database(embedding_model_path, vb_path, search_threshold, top_docs):
    # calling intialize_emb function to load the embedding model
    # EMBEDDING_INSTANCE.update_parameters(model_name=embedding_model_path)
    try:
        EMBEDDING_INSTANCE.update_parameters(model_name=embedding_model_path)
        # global EMBEDDING
        EMBEDDING = EMBEDDING_INSTANCE.embedding
        # getting data and intializing embedding function for query
        vectordb = Chroma(persist_directory=vb_path,
                          embedding_function=EMBEDDING)
        # getting vectordb as retriever
        print(vectordb)
        if vectordb.get() != []:
            print(vectordb)
            global retriever
            retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={
                                              "score_threshold": search_threshold, "k": top_docs})
            print(search_threshold, top_docs)
            # EMBEDDING.stop_instance()
            return retriever
        else:
            return "could not find embeddings generate it again."
    except Exception as e:
        return 'There is an issue {e}'


def get_answer(llm_model_path, embedding_model_path, vb_path, temp, search_threshold, top_docs, context_window, repetition_penality, top_k, top_p, user_question):
    # create the chain to answer questions
    # llm =chatbot(max_tokens,temp,repetition_penality,top_k,top_p,user_question)
    llm_model_path = llm_model_path
    embedding_model_path = embedding_model_path
    vb_path = vb_path
    temp = temp
    search_threshold = search_threshold
    top_docs = top_docs
    context_window = context_window
    repetition_penality = repetition_penality
    top_k = top_k
    top_p = top_p

    CHATBOT_INSTANCE.update_parameters(model=llm_model_path, temperature=temp, num_ctx=context_window,
                                       num_predict=128, repeat_penalty=repetition_penality, top_k=top_k, top_p=top_p)
    CHATBOT_LLM = CHATBOT_INSTANCE.llm
    without_raganswer = CHATBOT_LLM.invoke(user_question)
    global qa_chain
    # retriever=load_exsisting_database(embedding_model_path,vb_path,search_threshold,top_docs)
    print(retriever)
    prompt = ChatPromptTemplate.from_template("""Your main role is to provide accurate and concise responses using retrieved information. Consider the following key guidelines before answering questions.
Key Guidelines:
1. Use the retrieved Information to answer questions, ensure that your responses are relevant to the questions asked, If no relavent retrieved Information is provided to you ,say "I am unable to provide a suitable answer from the context provided."
2. If you dont know the answer, simply say that you dont know. Avoid providing incorrect or speculative information.
3. Ensure your responses are as human-like and conversational as possible.
5. If a question does not make any sense or is not factually coherent, explain why instead of providing incorrect information.
6. Neednot mention the context while answering the question
 <context>
{context}
</context>
Question: {input}""")
    print(prompt)
    combine_docs_chain = create_stuff_documents_chain(CHATBOT_LLM, prompt)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    answer = qa_chain.invoke({"input": user_question})
    raw_answer = answer['answer']
    print('raganswer', answer)
    EMBEDDING_INSTANCE.stop_instance()
    # CHATBOT_INSTANCE.invoke('/exit')
    return raw_answer, without_raganswer

# def model_loading(llm_model_path):
#     global model
#     global tokenizer
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
#         model = AutoModelForCausalLM.from_pretrained(llm_model_path,trust_remote_code=True).eval()
#     except Exception as e:
#         print(e)

# global CHATBOT
# CHATBOT = Chatbot(model=llm_model_path, temperature=temp, num_ctx=max_tokens,num_predict = 128,repeat_penalty = repetition_penality,top_k =top_k,top_p =top_p).llm


# global EMBEDDING
# EMBEDDING = EmbeddingsInitializer(model_name=embedding_model_path).embedding
