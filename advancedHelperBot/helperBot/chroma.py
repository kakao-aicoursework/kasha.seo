from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from . import fileLoader


CHROMA_PERSIST_DIR = "./upload"
CHROMA_COLLECTION_NAME = "bot_collection"
os.environ["OPENAI_API_KEY"] = fileLoader.read_text("./apiKey.txt")

def upload_embedding_from_file(file_path):
    documents = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".md") or file.endswith(".ipynb") or file.endswith(".txt"):
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)

upload_embeddings_from_dir("./assets/data")

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()

def getChroma():
    return _db

def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
