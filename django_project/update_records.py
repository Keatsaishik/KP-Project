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
from pprint import pprint

import oscom
import tiktoken
import datetime
import yaspin
import sys
import os
import time
import random
import openai
import time  # for measuring time duration of API calls
import os
import colorama
from colorama import Fore, Style
import shutil
from django.conf import settings
import openai_errors
import os
import json
from projectApp.models import Configuration



max_token_limit = 150000  # Set the maximum token limit
source_directory = ""
persist_directory = ""
model_name = "gpt-3.5-turbo"
chunk_size = 500
chunk_overlap = 50
openai.api_key = ""
embeddings = None

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
    print(f"source directory = {source_directory}")
    print(f"Persist directory = {persist_directory}")

def load_docs():
    loader = DirectoryLoader(source_directory)
    documents = loader.load()
    return documents


def split_docs(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs



# Define the storage dictionary to map file names to their records
file_records = {}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime.date, datetime.time)):
            return o.isoformat()  # Serialize date and time objects as ISO format strings
        return super().default(o)

def move_files():
    # Get the list of files in the source folder
    file_list = os.listdir(source_directory)

    for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(source_directory, file_name)

        if os.path.exists(file_path):
            # Remove each file present
            print(f'Removed {file_path}')
            os.remove(file_path)

            # Store the file record in the dictionary
            record = {
                'uploaded_date': datetime.date.today(),
                'uploaded_time': datetime.datetime.now().time(),
            }
            file_records[file_name] = record

    # Save the file records to a JSON file
    save_file_records()

def save_file_records():
    with open('storage_records.json', 'a') as storage_file:
        json.dump(file_records, storage_file, cls=CustomJSONEncoder)
        storage_file.write('\n')


def create_embeddings(docs):

    # Check if the vectordatabase exists or not
    if os.path.exists(persist_directory):
        # If exists, find it
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(" Exists persist ")
        
    else:
        # If does not exist, create it.
        os.makedirs(persist_directory)
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print(" Does not exist persist ")
        

    
    print("Hello")

    max_retries = 4  # Maximum number of retries
    retry_interval = 15  # Sleep duration in seconds
    for retry in range(max_retries):
        try:
            vectordb.add_documents(docs)
            vectordb.persist()
            return {'status': 500, 'message': "Upload Successful"}
        except openai.error.Timeout as e:
            # Handle timeout error
            error_message = f"OpenAI API request timed out: {str(e)}"
            return {'status': 400, 'message': error_message}
        except openai.error.APIError as e:
            # Handle API error
            error_message = f"OpenAI API returned an API Error: {str(e)}"
            return {'status': 401, 'message': error_message}
        except openai.error.APIConnectionError as e:
            # Handle connection error
            error_message = f"OpenAI API request failed to connect: {str(e)}"
            return {'status': 402, 'message': error_message}
        except openai.error.InvalidRequestError as e:
            # Handle invalid request error
            error_message = f"OpenAI API request was invalid: {str(e)}"
            return {'status': 403, 'message': error_message}
        except openai.error.AuthenticationError as e:
            # Handle authentication error
            error_message = f"OpenAI API request was not authorized: {str(e)}"
            return {'status': 404, 'message': error_message}
        except openai.error.PermissionError as e:
            # Handle permission error
            error_message = f"OpenAI API request was not permitted: {str(e)}"
            return {'status': 405, 'message': error_message}
        except openai.error.RateLimitError as e:
            # Handle rate limit error
            error_message = f"OpenAI API request exceeded rate limit: {str(e)}"
            if retry < max_retries - 1:
                print(f"Sleeping for {retry_interval}")
                time.sleep(retry_interval)
                retry_interval *= 2 
            else:
                return {'status': 406, 'message': error_message}

def get_token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    return num_tokens

def process_content(doc):
    text = doc.page_content + " \nsource_document = " + doc.metadata['source']
    doc.page_content = text
    return doc


def token_backoff():
    documents = load_docs()
    docs = split_docs(documents)
    token_count = 0
    docs_to_process = []
    
    for doc in docs:
        # Use this if you need to process the contents of the doc in a way 
        # that you need to modify your prompt
        # doc = process_content(doc)
        text = doc.page_content
        num_tokens = get_token_count(text)
        
        if token_count + num_tokens <= max_token_limit:
            # Add the doc to the list of docs to process
            docs_to_process.append(doc)
            token_count += num_tokens
        else:
            print(token_count)
            # Process the accumulated docs since the token limit is reached
            status_dict = create_embeddings(docs_to_process)
            # Print the statement if it exists
            if status_dict['status'] == 500:
                # That means it gave success, go for next chunk
                pass
            else : 
                # That means that reached some error
                return status_dict
            
                
                

            
            # Reset the token count and docs list, since it entered the else block so count the num_tokens for the last doc processed
            token_count = num_tokens
            docs_to_process = [doc]
    
    # Process any remaining docs after the loop ends
    # print(docs_to_process)

    status_dict = create_embeddings(docs_to_process)
    return status_dict

def main():
    load_directories()
    load_API_KEY()
    status_dict = token_backoff()
    if status_dict['status'] != 500:
        # That means that reached some error
        return status_dict
    # If the status is ok, move all the remaining files
    move_files()
    return status_dict

# if __name__ == "__main__":
#     main()
    

