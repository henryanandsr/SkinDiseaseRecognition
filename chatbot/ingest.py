from langchain.document_loaders import CSVLoader, UnstructuredHTMLLoader, TextLoader
import os

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
