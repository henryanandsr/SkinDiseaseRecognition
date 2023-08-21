from langchain.document_loaders import CSVLoader, UnstructuredHTMLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import argparse
import time
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

load_dotenv()

chunk_size = 512
chunk_overlap = 50

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

embeddings = HuggingFaceEmbeddings()

all_texts = []
html_folder_path = "../chatbot/db/additional_info/html"
# Iterate through HTML files in the folder
for html_file in os.listdir(html_folder_path):
    if html_file.endswith('.html'):
        html_file_path = os.path.join(html_folder_path, html_file)
        
        # Load and process HTML content
        loader = UnstructuredHTMLLoader(html_file_path, encoding='utf-8')
        documents2 = loader.load()
        texts2 = text_splitter.split_documents(documents2)
        all_texts.extend(texts2)

db = FAISS.from_documents(all_texts, embeddings)

def parse_arguments():
    parser = argparse.ArgumentParser(description='GPT: Ask questions to your documents without an internet connection, '
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
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

conversation_history = []

def main():
    mode = 0 # follow up question = 1, no = 0
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
            conversation_history = []


        start = time.time()

        answer = qa(context)['result']

        end = time.time()

        print(f"\n> Answer (took {round(end - start, 2)} s.):")

        if mode == 1 :
            conversation_history.append(answer)


if __name__ == '__main__':
    main()