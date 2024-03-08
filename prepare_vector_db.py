from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

document_file_path = "data"

DB_FAISS_PATH = 'vectorstorepdf/db_faiss'

def createDB():
    loader = DirectoryLoader(document_file_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 50)

    chunks = text_splitter.split_documents(documents)

    #Embedding
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(chunks, embedding_model)

    db.save_local(DB_FAISS_PATH)
    
    return db

createDB()
