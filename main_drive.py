import os
import streamlit as st
import gdown
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
import shutil

# Google Drive Datei-IDs 
FAISS_INDEX_ID = "1jSBoE_P9TNCqePGBuymHyoYwO2lMaCV0"
FAISS_PKL_ID = "1qzw7VadNOJfoO34Hsp_l39XDQfKvm0Fk"

# Verzeichnisse erstellen
FAISS_PATH = "faiss_db"
os.makedirs(FAISS_PATH, exist_ok=True)

# Dateien aus Google Drive herunterladen
def download_faiss():
    gdown.download(f"https://drive.google.com/uc?id={FAISS_INDEX_ID}", f"{FAISS_PATH}/index.faiss", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={FAISS_PKL_ID}", f"{FAISS_PATH}/index.pkl", quiet=False)


# Initialisierung der FAISS-Datenbank
def load_faiss_database():
    try:
        vector_store = FAISS.load_local(
            "faiss_database", 
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Fehler beim Laden der FAISS-Datenbank: {e}")
        st.stop()

# Erstellung des RAG-Chatbots
def create_rag_chatbot():
    vector_store = load_faiss_database()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"]),
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Streamlit-UI
def main():
    st.set_page_config(page_title="IFRS-Chatbot mit RAG", page_icon="🤖")
    st.title("🤖 Chatbot mit RAG-Funktion. Dieser Chatbot beantwortet Fragen zu den derzeit in der EU gültigen IFRS-Standards.")

    st.sidebar.header("Einstellungen")
    reset_chat = st.sidebar.button("Chat zurücksetzen")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = create_rag_chatbot()

    if reset_chat:
        st.session_state.chat_history = []  # Reset chat history
        st.rerun()

    # Chat-Verlauf anzeigen mit st.chat_message
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message[0])
        with st.chat_message("assistant"):
            st.markdown(message[1])

    # Eingabefeld mit st.chat_input
    user_input = st.chat_input("Stelle eine Frage:")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Antwort wird generiert..."):
            try:
                response = st.session_state.qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
                st.session_state.chat_history.append((user_input, response["answer"]))

                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {e}")

if __name__ == "__main__":
    main()