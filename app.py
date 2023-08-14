import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_chat import message
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatOpenAI
import qdrant_client
import os
import random
import itertools

@st.cache_data()
def generate_eval(text, N, chunk):
    n = len(text)
    if n != 0:
      starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
      sub_sequences = [text[i:i+chunk] for i in starting_indices]
      chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
      eval_set = []
      for i, b in enumerate(sub_sequences):
          try:
              qa = chain.run(b)
              eval_set.append(qa)
          except:
              pass
      eval_set_full = list(itertools.chain.from_iterable(eval_set))
      return eval_set_full

@st.cache_data()
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

@st.cache_resource
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vectorstore(text_chunks):
    st.session_state.client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    st.session_state.vectors_config = qdrant_client.http.models.VectorParams(
        size=1536, distance=qdrant_client.http.models.Distance.COSINE)

    st.session_state.client.recreate_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config=st.session_state.vectors_config
    )
    embeddings = OpenAIEmbeddings()

    vectorstore = Qdrant(
        client=st.session_state.client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings
    )
    vectorstore.add_texts(text_chunks)
    return vectorstore

@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain

@st.cache_data(persist=True)
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

def clicked(button):
    st.session_state.clicked[button] = True

def main():
    load_dotenv()
    # Add custom CSS

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "clicked" not in st.session_state:
        st.session_state.clicked = {1: False, 2: False}

    if "selected" not in st.session_state:
        st.session_state.selected = "PDF GPT"

    if "eval_set" not in st.session_state:
        st.session_state.eval_set = None

    st.set_page_config(page_title="PDF GPT", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
            .css-card {
                border-radius: 0px;
                padding: 30px 10px 10px 10px;
                background-color: #f8f9fa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 10px;
                font-family: "IBM Plex Sans", sans-serif;
            }
            
            .card-tag {
                border-radius: 0px;
                padding: 1px 5px 1px 5px;
                margin-bottom: 10px;
                position: absolute;
                left: 0px;
                top: 0px;
                font-size: 0.6rem;
                font-family: "IBM Plex Sans", sans-serif;
                color: white;
                background-color: #D2042D;
                }
                
            .css-zt5igj {left:0;
            }
            
            span.css-10trblm {margin-left:0;
            }
            
            div.css-1kyxreq {margin-top: -40px;
            }
          
        </style>
        """,
        unsafe_allow_html=True,
    )

    colored_header(
          label="Ask questions about your PDFs - PDF GPT :books:",
          description=None,
          color_name="red-70"
      )

    st.session_state.user_question = st.chat_input("Ask your question here...")
    st.info('If you want to reset, just reload the page or remove your PDFs😉', icon="ℹ️")

    if st.session_state.user_question:
        handle_userinput(st.session_state.user_question)

    with st.sidebar:
      st.subheader("Your documents 📄")

      pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process ⬆️", accept_multiple_files=True)
        
      st.button('Process 🔍', on_click=clicked, args=[1])

      if st.session_state.clicked[1]:
            with st.spinner("Processing..."):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chuncks
                text_chunks = get_text_chunks(raw_text)

                # Create vector storage
                vectorstore = get_vectorstore(text_chunks)

                # Conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                

                st.sidebar.subheader("Auto-Generated Review Questions:")

                st.session_state.eval_set = generate_eval(raw_text, N=5, chunk=3000)

                for qa_pair in st.session_state.eval_set:
                  st.markdown(
                    f"""
                    <div class="css-card">
                      <span class="card-tag">Question</span>
                      <p style="font-size: 12px; color: black;">{qa_pair['question']}</p>
                      <p style="font-size: 12px; color: black;">{qa_pair['answer']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                  )

                st.success("Successfully Processed your PDFs!")
                st.balloons() 

if __name__ == '__main__':
  try:
    main()

  except:
    st.error("Oops! Something went wrong. Please try again later. 😅")

