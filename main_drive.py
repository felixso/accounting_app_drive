import streamlit as st
import os
import tempfile
import gdown
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Streamlit App-Titel setzen
st.title("RAG Chatbot mit FAISS-Datenbank")

# API-Key für Groq aus Streamlit Secrets oder Umgebungsvariablen
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

if not groq_api_key:
    st.error("Kein GROQ API-Key gefunden. Bitte stellen Sie sicher, dass der API-Key in den Streamlit-Secrets oder als Umgebungsvariable gesetzt ist.")
    st.stop()

# URLs für FAISS-Dateien auf Google Drive
faiss_index_url = st.secrets.get("FAISS_INDEX_URL", os.environ.get("FAISS_INDEX_URL"))
faiss_pkl_url = st.secrets.get("FAISS_PKL_URL", os.environ.get("FAISS_PKL_URL"))

if not faiss_index_url or not faiss_pkl_url:
    st.error("URLs für FAISS-Dateien fehlen. Bitte stellen Sie sicher, dass diese in den Streamlit-Secrets oder als Umgebungsvariablen gesetzt sind.")
    st.stop()

# Funktion zum Herunterladen der FAISS-Dateien von Google Drive
@st.cache_resource
def load_faiss_from_drive():
    st.info("FAISS-Datenbank wird geladen...")
    temp_dir = tempfile.mkdtemp()
    
    # FAISS-Index herunterladen
    faiss_index_path = os.path.join(temp_dir, "index.faiss")
    gdown.download(faiss_index_url, faiss_index_path, quiet=False)
    
    # FAISS-PKL herunterladen
    faiss_pkl_path = os.path.join(temp_dir, "index.pkl")
    gdown.download(faiss_pkl_url, faiss_pkl_path, quiet=False)
    
    # HuggingFace Embeddings initialisieren
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # FAISS-Datenbank laden
    vectorstore = FAISS.load_local(temp_dir, embeddings, "index",allow_dangerous_deserialization=True)
    st.success("FAISS-Datenbank erfolgreich geladen!")
    
    return vectorstore

# FAISS-Datenbank laden
try:
    vectorstore = load_faiss_from_drive()
except Exception as e:
    st.error(f"Fehler beim Laden der FAISS-Datenbank: {e}")
    st.stop()

# Groq LLM initialisieren
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",  # Kann angepasst werden
    temperature=0.5,
)

# Gesprächsverlauf im Session State initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Gesprächsspeicher für Langchain initialisieren
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Konversationskette erstellen
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    verbose=True,
)

# Bisherigen Chatverlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Benutzereingabe
user_query = st.chat_input("Stellen Sie Ihre Frage...")

if user_query:
    # Benutzernachricht anzeigen
    with st.chat_message("user"):
        st.write(user_query)
    
    # Nachricht zum Chatverlauf hinzufügen
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Antwort generieren
    with st.spinner("Antwort wird generiert..."):
        try:
            response = qa_chain({"question": user_query})
            ai_response = response["answer"]
            
            # Quellen aus Dokumenten extrahieren, wenn vorhanden
            if "source_documents" in response and response["source_documents"]:
                sources = set()
                for doc in response["source_documents"]:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.add(doc.metadata["source"])
                
                if sources:
                    ai_response += "\n\n**Quellen:**\n"
                    for i, source in enumerate(sources, 1):
                        ai_response += f"{i}. {source}\n"
        
        except Exception as e:
            ai_response = f"Bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten: {str(e)}"
    
    # Antwort anzeigen
    with st.chat_message("assistant"):
        st.write(ai_response)
    
    # Antwort zum Chatverlauf hinzufügen
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Seitenleiste mit Informationen
with st.sidebar:
    st.title("Über diesen Chatbot")
    st.write("""
    Dieser Chatbot nutzt RAG (Retrieval-Augmented Generation) mit einer FAISS-Datenbank, 
    um Antworten auf Ihre Fragen zu generieren.
    
    **Verwendete Technologien:**
    - Streamlit für die Benutzeroberfläche
    - Groq LLM für die Textgenerierung
    - Langchain für die RAG-Pipeline
    - HuggingFace Embeddings für die Vektorisierung
    - FAISS als Vektordatenbank
    """)
    
    # Optionale Benutzereinstellungen
    st.subheader("Einstellungen")
    temperature = st.slider("Kreativität (Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if temperature != 0.5:
        llm.temperature = temperature
        st.info(f"Temperature wurde auf {temperature} gesetzt.")
        
    # Optionen zum Zurücksetzen des Chatverlaufs
    if st.button("Chatverlauf zurücksetzen"):
        st.session_state.messages = []
        memory.clear()
        st.success("Chatverlauf wurde zurückgesetzt!")
        st.experimental_rerun()
