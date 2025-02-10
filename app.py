import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama 
from langchain.memory import ConversationBufferMemory
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate) 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough
import faiss
from htmlTemplates import css, user_template, bot_template
from dotenv import load_dotenv


model = ChatOllama(model='llama3.2:1b', base_url='http://localhost:11434')

def inject_custom_css():
    
    with open(".venv\style.css") as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            
            # Check if the PDF is encrypted and attempt to decrypt it
            if pdf_reader.is_encrypted:
                try:
                    # Assuming you know the password, otherwise, you'd need to provide it
                    pdf_reader.decrypt("")  # Pass an empty string for no password, or provide the actual password
                except Exception as e:
                    print(f"Failed to decrypt {pdf}: {e}")
                    continue  # Skip this PDF and move to the next one if decryption fails

            for page in pdf_reader.pages:
                text += page.extract_text()

        except Exception as e:
            print(f"Error reading {pdf}: {e}")
            continue  # Skip this PDF and continue with the next one

    return text

def get_text_chunks(text):
    if not text:
        raise ValueError("Input text is empty or None!")
    
    text_splitter = CharacterTextSplitter(
        separator="\n",       # Splits text on newlines
        chunk_size=1000,      # Maximum chunk size (characters)
        chunk_overlap=200,    # Number of characters to overlap between chunks
        length_function=len   # Function to calculate text length
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Optional: Filter out empty chunks, if any
    chunks = [chunk for chunk in chunks if chunk.strip()]

    # If no valid chunks are found, raise an error
    if not chunks:
        raise ValueError("No valid text chunks were created.")

    return chunks

def get_vectorstore(text_chunks):
    # Check if text_chunks is empty
    if not text_chunks:
        raise ValueError("Text chunks cannot be empty!")

    # Generate embeddings using OllamaEmbeddings
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

    # Ensure that the embeddings are generated correctly and are not empty
    try:
        vectorstore = FAISS.from_texts(
            texts=text_chunks, 
            embedding=embeddings, 
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    except Exception as e:
        # Catch any exceptions that occur during vector store creation and raise an informative error
        raise ValueError(f"Error creating vector store: {e}")
    
    return vectorstore

def get_conversation(vectorstore):
    llm = model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_userinput(user_question):
    
    prompt = """
    You are an assistant for question-answering tasks. Use the following context to answer the question. 
    Answer the question in detail using information only from the context provided. If possible answer in bullet points.
    Context: {context}
    Question: {question} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt)
    
    response = st.session_state.conversation({'question' : user_question})
    
    st.session_state.chat_history = response['chat_history']
       
    # This function will allows us to loop through the chat history
    for i, message in enumerate(st.session_state.chat_history):
        #  The modulo '%' allows us to target the odd numbered chats in the history, which as the ones that belong to the user and store those in the chat history
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


load_dotenv()    
def main():
    
    st.set_page_config(page_title='Chat with multiple PDFs', page_icon=":books")
    
    inject_custom_css()
    
    st.header(':speech_balloon: Chat with multiple PDFs :books:')
    user_question = st.chat_input("Ask a question about your documents:")
    
        
    st.write(bot_template.replace("{{MSG}}", "Hello NT Offical, how can I assist you today?"), unsafe_allow_html=True)
            
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
       
    if user_question:
        with st.spinner('typing response......'):
            handle_userinput(user_question)
    
                 
    # Creating a button to reset, clear conversation and history
    add_vertical_space(2)
    reset_button_key = "reset_button"
    reset_button = st.button("Reset Chat",key=reset_button_key)
    if reset_button:
        st.session_state.conversation = None
        st.session_state.chat_history = None
    
    with st.sidebar:
        st.subheader("Your documents :open_file_folder:")
        add_vertical_space(1)
        
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                                                
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)              
                
                # Create the vector store with embbedings
                vectorstore = get_vectorstore(text_chunks)
                
                # Create converstaion chain/memory
                st.session_state.conversation = get_conversation(vectorstore)
            
            st.success('Processing Complete!')
        
        
        add_vertical_space(15)
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io)
        - [Langchain](https://python.langchain.com/)
        - [ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/) LLM model
        
        ''')
        
        add_vertical_space(1)
        st.write('Created by [Sandile S Dube](https://linkedin.com/in/sandile-dube-9b7a5431) :earth_africa: :male-artist:')
                          

if __name__ == "__main__":
    main()