from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


import sys
import os
import time
from yaspin import yaspin
import json
from pprint import pprint
import random
import openai
import time  # for measuring time duration of API calls
import os
import shutil
import colorama
from colorama import Fore, Style
import shutil

from threading import Thread
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from langchain.llms import OpenAI
from langchain.callbacks.base import BaseCallbackHandler
from django.conf import settings
from queue_module import result_queue

from projectApp.models import Configuration


source_directory = ""
# destination_directory = "Data Storage"
persist_directory = ""
model_name = "gpt-3.5-turbo"
chunk_size = 800
chunk_overlap = 50
openai.api_key = ""
embeddings = None

q = Queue()

def load_directories():
    global source_directory, persist_directory
    config = Configuration.objects.get()
    database_name = config.database_directory 
    source_directory = os.path.join(settings.DATABASE_DIRECTORY, database_name, settings.UPLOAD_NAME)
    persist_directory = os.path.join(settings.DATABASE_DIRECTORY, database_name, settings.INDEX_NAME)

def load_API_KEY():
    global embeddings
    import security
    openai.api_key = security.API_KEY
    os.environ['OPENAI_API_KEY'] = security.API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


def stream(retrieval_chain, query):
    
    # Create a Queue
    
    job_done = object()
    

    def task():
        result = retrieval_chain({"query":query}, return_only_outputs=True)  
        result_queue.put(result)  # Store the response in the shared object
        q.put(job_done)
    
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token
            # yield str(next_token).decode('utf-8')
        except Empty:
            continue
    
    
    result = result_queue.get()
    paths = []
    source_document = result["source_documents"]
    for count, sr in enumerate(source_document):
        content = source_document[count].page_content
        metadata = source_document[count].metadata
        paths.append(metadata)
        
        
    file_names = [os.path.basename(path['source']) for path in paths]
    yield "/1234567890@3##5733##"

    yield "\nThe source documents are:\n"
    for file_name in file_names:
        yield file_name + "\n"

    # if 'memory' not in request.session:
        #     # Compute retrieval_chain if it's not already stored in session
        #     request.session["memory"] = retrieval_chain.memory            
        # else:
        #     print(f" Memory = {request.session['memory']}")
        #     retrieval_chain.memory = request.session["memory"] 

        

def get_retrieval_chain(vectordb):
    llm = ChatOpenAI(streaming=True, callbacks=[QueueCallback(q)], temperature=0)
    PROMPT = get_prompts()
    retrieval_chain = RetrievalQA.from_chain_type(
                        llm,
                        chain_type="stuff",
                        retriever=vectordb.as_retriever(),
                        return_source_documents=True,
                        chain_type_kwargs={
                            "verbose": True,
                            "prompt": PROMPT,
                            "memory": ConversationBufferMemory(
                                memory_key="history",
                                input_key="question"),
                        }
                )


    print(" Thank You! ")
    return retrieval_chain

def initialise_llm():

    # Check if index exists
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return get_retrieval_chain(vectordb)
    else:
        print(" No records exist, please update records by adding new files ")
        # Handle exception or errors
   
    




def get_prompts():
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    You are an helpful assistant for Kolkata Police. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    return prompt
    
    

def get_answer():
    
    # query = input("Enter your query : ")
    load_directories()
    load_API_KEY()
    retrieval_chain = initialise_llm()
    return retrieval_chain
    


