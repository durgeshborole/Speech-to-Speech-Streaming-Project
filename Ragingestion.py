from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    #1.Load Documents
    print("Loading Documents...")
    loader = TextLoader(r"C:\Users\Deepak Borole\PycharmProjects\pythonProject\Infosys internship\Rammandir.txt")
    document= loader.load()
    print(f"Loaded{ len(document)}")

    #splitting
    print("Splitting Documents")
    splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split_documents=splitter.split_documents(document)
    print(f"Split{len(document)} documents into {(len(split_documents))}")

    #3. Embedding Document
    print("starting embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #4.Inserting Documents into VectorDb
    print("Inserting Documents into VectorDb")
    vector_db = PineconeVectorStore.from_documents(split_documents,embeddings,index_name = os.getenv("INDEX_NAME"))
    print(f"Inserted {len(split_documents)} documents into VectorDb")
    
