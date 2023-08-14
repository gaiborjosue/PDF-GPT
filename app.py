import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
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
    embeddings = embeddings
  )

  vectorstore.add_texts(text_chunks)

  return vectorstore

def get_conversation_chain(vectorstore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
  )

  return conversation_chain

def handle_userinput(user_question):
  response = st.session_state.conversation({'question': user_question})
  st.session_state.chat_history = response['chat_history']

  for i, mes in enumerate(st.session_state.chat_history):
    if i % 2 == 0:
      with st.chat_message("user"):
        st.write(mes.content)
    else:
      with st.chat_message("assistant"):
        st.write(mes.content)

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

  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  if "user_question" not in st.session_state:
    st.session_state.user_question = None

  st.set_page_config(page_title="PDF GPT", page_icon=":books:")
  st.write(css, unsafe_allow_html=True)
  st.header("Ask questions about your PDFs - PDF GPT :books:")


  st.session_state.user_question = st.chat_input("Ask your question here...")
  st.info('If you want to reset, just reload the page 😉', icon="ℹ️")
  if st.session_state.user_question:
    handle_userinput(st.session_state.user_question)

  with st.sidebar:
     st.subheader("Your documents 📄")

     pdf_docs = st.file_uploader("Upload your PDFs here and click on Process ⬆️", accept_multiple_files=True)

     if st.button("Process 🔍"):
      with st.spinner("Processing..."):
        # Get pdf text
        raw_text = get_pdf_text(pdf_docs)
      
        # Get text chuncks
        text_chunks = get_text_chunks(raw_text)

        # Create vector storage
        vectorstore = get_vectorstore(text_chunks, client)

        # Conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.success("Successfully Processed your PDFs!")
        st.balloons()


    
if __name__ == '__main__':
    main()