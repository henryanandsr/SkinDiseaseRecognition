from langchain.document_loaders import CSVLoader, UnstructuredHTMLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import argparse
import pandas as pd
import time
import pyarrow.feather as feather
import faiss
import numpy as np
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv()

chunk_size = 500
chunk_overlap = 50

# Specify the output CSV file path
csv_file_path = '../chatbot/db/diagnose_en_dataset.csv'
# Check if the CSV file already exists
if not os.path.exists(csv_file_path):
    df = feather.read_feather("../chatbot/db/diagnose_en_dataset.feather")
    df.to_csv(csv_file_path, index=False,encoding='utf-8')
    print(f"CSV file '{csv_file_path}' created successfully.")

loader = CSVLoader(csv_file_path,encoding='utf-8')
documents = loader.load()
documents = documents[:200]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

db_list = []
html_folder_path = "../chatbot/db/additional_info/html"
# Iterate through HTML files in the folder
for html_file in os.listdir(html_folder_path):
    if html_file.endswith('.html'):
        html_file_path = os.path.join(html_folder_path, html_file)
        
        # Load and process HTML content
        loader = UnstructuredHTMLLoader(html_file_path, encoding='utf-8')
        documents2 = loader.load()
        texts2 = text_splitter.split_documents(documents2)
        
        # Generate embeddings and build FAISS index
        db2 = FAISS.from_documents(texts2, embeddings)
        
        # Store the FAISS index for this HTML file
        db_list.append(db2)
        
# html_file_path = '../embedding/db/additional_info/html/Skin Diseases - Disease Control Priorities in Developing Countries - NCBI Bookshelf.html'
# loader = UnstructuredHTMLLoader(html_file_path,encoding='utf-8')
# documents2 = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
# texts2 = text_splitter.split_documents(documents2)

db2 = FAISS.from_documents(texts2, embeddings)

# 2. Function for similarity search

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

# def retrieve_info2(query):
#     similar_response = db_list.similarity_search(query, k=3)

#     page_contents_array = [doc.page_content for doc in similar_response]

#     # print(page_contents_array)

#     return page_contents_array

def retrieve_info2(query):
    page_contents_array = []

    # Iterate through all FAISS indexes in db_list
    for db2 in db_list:
        similar_response = db2.similarity_search(query, k=3)
        page_contents = [doc.page_content for doc in similar_response]
        page_contents_array.extend(page_contents)

    return page_contents_array


# 3. Setup LLMChain & prompts

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
args = parse_arguments()
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

template = """
You are a medical assistant with expert dermatology knowledge. You approach conversations with empathy and positivity.
I'll share a patient's message, and you'll provide the best response based on past additional_knowledges. 
Please follow these guidelines:

1/ Response should be focus on accurate facts from additional informations to answer message

2/ Response should be similar to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

3/ If the best practice are irrelevant, then try to mimic the style of the best practice to {name}'s message

4/ Respond as if you were having a natural conversation.

Below is a message I received from the {name}:
{message}

Here is a list of additional knowledges about dermatology. Please focus on these additionall knowledges. Use this knowledges for more accurate response:
{additional_knowledge}

Here is a list of how we behave in respond to a message:
{best_practice}

Please write the best response with most accurate facts that I should send to {name}:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice", "name", "additional_knowledge"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message, user_name):
    best_practice = retrieve_info(message)
    additional_knowledge = retrieve_info2(message)
    response = chain.run(message=message, best_practice=best_practice, name=user_name, additional_knowledge=additional_knowledge)
    return response

conversation_history = []

# 5. Build an app with streamlit
def main():
    mode = 0
    count = 0

    # Greetings to start the conversation
    print("Hello! I'm Michie, your friendly and empathetic medical assistant. How can I assist you today?")
    print("Feel free to ask anything about skin diseases or type 'exit' to end.")

    user_name = input("\nBefore we continue, may I know your name? ")

    # Interactive questions and answers
    while True:
        query = input(f"\n{user_name}: ")
        if query == "exit":
            print("Thank you for using DermaticaAI. Take care!")
            break
        if query.strip() == "":
            continue
        
        if mode == 1 :
            count+=1

            # Add the query to the conversation history
            conversation_history.append(query)

            # Concatenate conversation history to provide context (for checking context)
            context = "\n".join(conversation_history)

            if conversation_history and count >= 3 :
                conversation_history.pop(0)
                conversation_history.pop(0)
            
            # Concatenate conversation history to provide context (for answering)
            context = "\n".join(conversation_history)
        else :
            context = query

        # print("---------------context-------------")
        # print(context)
        # print("-------------end-------------")

        start = time.time()

        answer= generate_response(context, user_name)

        end = time.time()

        print(f"\n> Answer (took {round(end - start, 2)} s.):")

        if mode == 1 :
            conversation_history.append(answer)


if __name__ == '__main__':
    main()