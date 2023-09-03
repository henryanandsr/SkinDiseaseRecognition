from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest import ingest_data
from qa import retrieval_qa
import time
import os
from dotenv import load_dotenv
from google.auth import impersonated_credentials
from google.oauth2 import service_account

app = Flask(__name__)


# Impersonate service account
key_path = 'dermaticaai-397114-1a63cb8685a8.json'
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

load_dotenv()

def remove_bug(answer) :

    # remove prefix
    prefixes_to_remove = ["AI:", "Human:"]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip()
    
    # remove whitespace char in front
    answer = answer.lstrip()

    # split with dot
    answers = answer.split('.')

    # remove defects in the back
    if len(answers) > 1:
        final_answer = '.'.join(answers[:-1])

    # error message
    elif answer == ".":
        final_answer = "I'm sorry, I encountered an error while processing your request. Please make a new chat. Thank you!"
    
    # normal message
    else :
        final_answer = answer

    final_answer = final_answer + '.'
    
    return final_answer

def main():

    

    chunk_size = 512
    chunk_overlap = 50

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = HuggingFaceEmbeddings()

    txt_folder_path = "db/additional_info/txt"
    all_texts = ingest_data(txt_folder_path, text_splitter)

    db = FAISS.from_documents(all_texts, embeddings)

    qa = retrieval_qa(db)

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

        start = time.time()

        answer = qa(query)['result']

        final_answer = remove_bug(answer) #final answer/response

        print(final_answer)

        end = time.time()


        print(f"\n> Answer (took {round(end - start, 2)} s.):")

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        query = data['query']
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400

        start = time.time()
        answer = qa(query)['result']
        final_answer = remove_bug(answer)  # Final answer/response
        end = time.time()

        response = {
            'answer': final_answer,
            'response_time': round(end - start, 2)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_dotenv()
    chunk_size = 512
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = HuggingFaceEmbeddings()
    txt_folder_path = "db/additional_info/txt"
    all_texts = ingest_data(txt_folder_path, text_splitter)
    db = FAISS.from_documents(all_texts, embeddings)
    qa = retrieval_qa(db)
    app.run(host='0.0.0.0', port=5000)