import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 📥 Step 1: Load multiple PDFs
def load_pdf_files(file_paths):
    all_documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_documents.extend(docs)
    return all_documents

# 📄 Specify your PDF files here
pdf_files = [
  r"C:\Users\Devang Varshney\OneDrive\Desktop\finutunning\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND (1).pdf",
    r"C:\Users\Devang Varshney\OneDrive\Desktop\finutunning\parks-preventive-social-medicine-23rd-ed.pdf",
    r"C:\Users\Devang Varshney\OneDrive\Desktop\finutunning\Guyton-and-Hall-Textbook-of-Medical-Physiology-12th-Ed (1).pdf",
    r"C:\Users\Devang Varshney\OneDrive\Desktop\finutunning\Atlas of Human Anatomy, Sixth Edition- Frank H. Netter, M.D.pdf"
]


documents = load_pdf_files(pdf_files)
#print(len(documents))

# ✂️ Step 2: Chunk the documents
def create_chunks(extracted_data):
    all_text = " ".join([doc.page_content for doc in extracted_data])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(all_text)
    final_chunks = [Document(page_content=chunk) for chunk in text_chunks]
    return final_chunks

test_chunks = create_chunks(documents)
#print(f"Total chunks created: {len(test_chunks)}")
#rint(test_chunks[0].page_content[:300])
# 🧠 Step 3: Get embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# 📦 Step 4: Save to or update FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"

if os.path.exists(DB_FAISS_PATH):
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    db.add_documents(test_chunks)
else:
    db = FAISS.from_documents(test_chunks, embedding_model)

db.save_local(DB_FAISS_PATH)