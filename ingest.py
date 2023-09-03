from langchain.document_loaders import TextLoader
import os

def ingest_data(txt_folder_path, text_splitter) :
    all_texts = []
    # Iterate through txt files in the folder
    for txt_file in os.listdir(txt_folder_path):
        if txt_file.endswith('.txt'):
            txt_file_path = os.path.join(txt_folder_path, txt_file)
            
            # Load and process txt content
            loader = TextLoader(txt_file_path, encoding='utf-8')
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            all_texts.extend(texts)
    return all_texts
