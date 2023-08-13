import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import qdrant_client
import os


def get_pdf_text(pdf_docs):
  text = ""

  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()

  return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)
  return chunks

def get_vectorstore(text_chunks, client):
  embeddings = OpenAIEmbeddings()

  vectorstore=Qdrant(
    client=client,
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
  )

  return vectorstore

def main():
  load_dotenv()

  client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
  )
  vectors_config = qdrant_client.http.models.VectorParams(size=1536, distance= qdrant_client.http.models.Distance.COSINE)

  client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config = vectors_config
  )



  st.set_page_config(page_title="PDF GPT", page_icon=":books:")
  st.header("Ask questions about your PDFs - PDF GPT")
  st.text_input("Ask a question about your documents:")
  
  with st.sidebar:
     st.subheader("Your documents")
     pdf_docs = st.file_uploader("Upload your PDFs here and click on Process", accept_multiple_files=True)
     if st.button("Process"):
      with st.spinner("Processing"):
        # Get pdf text
        raw_text = get_pdf_text(pdf_docs)
      
        # Get text chuncks
        text_chunks = get_text_chunks(raw_text)

        # Create vector storage
        vectorstore = get_vectorstore(text_chunks)


    
if __name__ == '__main__':
    main()