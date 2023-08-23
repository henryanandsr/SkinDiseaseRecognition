from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest import ingest_data
from qa import retrieval_qa
import time
from dotenv import load_dotenv

load_dotenv()
# html_folder_path = "../chatbot/db/additional_info/html"
# # Iterate through HTML files in the folder
# for html_file in os.listdir(html_folder_path):
#     if html_file.endswith('.html'):
#         html_file_path = os.path.join(html_folder_path, html_file)
        
#         # Load and process HTML content
#         loader = UnstructuredHTMLLoader(html_file_path, encoding='utf-8')
#         documents2 = loader.load()
#         texts2 = text_splitter.split_documents(documents2)
#         all_texts.extend(texts2)

def main():

    chunk_size = 512
    chunk_overlap = 50

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = HuggingFaceEmbeddings()

    txt_folder_path = "../chatbot/db/additional_info/txt"
    all_texts = ingest_data(txt_folder_path, text_splitter)

    db = FAISS.from_documents(all_texts, embeddings)

    qa = retrieval_qa(db)

    conversation_history = []
    mode = 1 # follow up question = 1, no = 0
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