from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest import ingest_data
from qa import retrieval_qa
import time
from dotenv import load_dotenv

load_dotenv()

def remove_bug(answer) :
    prefixes_to_remove = ["AI:", "Human:"]
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip()
    answer = answer.lstrip()
    answers = answer.split('.')
    if len(answers) > 1:
        final_answer = '.'.join(answers[:-1])
    else :
        final_answer = answer
    final_answer = final_answer + '.'
    return final_answer

def main():

    chunk_size = 512
    chunk_overlap = 50

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = HuggingFaceEmbeddings()

    txt_folder_path = "../chatbot/db/additional_info/txt"
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

        print("----------final answer------------")
        print(final_answer)
        print("----------------------")

        end = time.time()


        print(f"\n> Answer (took {round(end - start, 2)} s.):")


if __name__ == '__main__':
    main()