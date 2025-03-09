import streamlit as st
import os
import tempfile
import requests
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

# Google Drive Datei-IDs
faiss_index_id = st.secrets.get("FAISS_INDEX_ID", os.environ.get("FAISS_INDEX_ID"))
faiss_pkl_id = st.secrets.get("FAISS_PKL_ID", os.environ.get("FAISS_PKL_ID"))

if not faiss_index_id or not faiss_pkl_id:
    st.error("Google Drive File-IDs für FAISS-Dateien fehlen. Bitte stellen Sie sicher, dass diese in den Streamlit-Secrets oder als Umgebungsvariablen gesetzt sind.")
    st.stop()

# Funktion zum Herunterladen von Google Drive mit direktem Download-Link
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Für große Dateien benötigen wir manchmal einen Bestätigungstoken
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Prüfen, ob es ein Bestätigungsformular gibt
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        url = f"{url}&confirm={token}"
        response = session.get(url, stream=True)
    
    # Datei speichern
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# Funktion zum Laden der FAISS-Datenbank von Google Drive
@st.cache_resource
def load_faiss_from_drive():
    st.info("FAISS-Datenbank wird geladen...")
    temp_dir = tempfile.mkdtemp()
    
    # FAISS-Index herunterladen
    faiss_index_path = os.path.join(temp_dir, "index.faiss")
    st.text("Downloading index.faiss...")
    download_file_from_google_drive(faiss_index_id, faiss_index_path)
    
    # FAISS-PKL herunterladen
    faiss_pkl_path = os.path.join(temp_dir, "index.pkl")
    st.text("Downloading index.pkl...")
    download_file_from_google_drive(faiss_pkl_id, faiss_pkl_path)
    
    # HuggingFace Embeddings initialisieren
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # FAISS-Datenbank laden
    st.text("Loading FAISS vector store...")
    vectorstore = FAISS.load_local(temp_dir, embeddings, "index")
    st.success("FAISS-Datenbank erfolgreich geladen!")
    
    return vectorstore

# Hilfsfunktion zur Validierung der FAISS-Dateien
def validate_files(temp_dir):
    index_path = os.path.join(temp_dir, "index.faiss")
    pkl_path = os.path.join(temp_dir, "index.pkl")
    
    st.text(f"Überprüfe Dateien im Verzeichnis {temp_dir}:")
    
    if os.path.exists(index_path):
        file_size = os.path.getsize(index_path)
        st.text(f"- index.faiss gefunden: {file_size} Bytes")
        
        # Prüfen, ob es sich um eine HTML-Datei handelt
        with open(index_path, 'rb') as f:
            header = f.read(10).decode('latin-1', errors='ignore')
            if header.startswith('<!DOCTYPE') or header.startswith('<html') or header.startswith('<!DO'):
                st.error("Die index.faiss-Datei enthält HTML statt der erwarteten FAISS-Daten!")
                return False
    else:
        st.error("index.faiss nicht gefunden!")
        return False
        
    if os.path.exists(pkl_path):
        file_size = os.path.getsize(pkl_path)
        st.text(f"- index.pkl gefunden: {file_size} Bytes")
        
        # Prüfen, ob es sich um eine HTML-Datei handelt
        with open(pkl_path, 'rb') as f:
            header = f.read(10).decode('latin-1', errors='ignore')
            if header.startswith('<!DOCTYPE') or header.startswith('<html') or header.startswith('<!DO'):
                st.error("Die index.pkl-Datei enthält HTML statt der erwarteten Pickle-Daten!")
                return False
    else:
        st.error("index.pkl nicht gefunden!")
        return False
        
    return True

# FAISS-Datenbank laden
try:
    # Wenn Debug-Option aktiviert ist, detaillierte Informationen anzeigen
    debug_mode = st.sidebar.checkbox("Debug-Modus")
    
    if debug_mode:
        temp_dir = tempfile.mkdtemp()
        st.info(f"Temporäres Verzeichnis: {temp_dir}")
        
        # FAISS-Index herunterladen
        faiss_index_path = os.path.join(temp_dir, "index.faiss")
        st.text(f"Downloading index.faiss (ID: {faiss_index_id})...")
        download_file_from_google_drive(faiss_index_id, faiss_index_path)
        
        # FAISS-PKL herunterladen
        faiss_pkl_path = os.path.join(temp_dir, "index.pkl")
        st.text(f"Downloading index.pkl (ID: {faiss_pkl_id})...")
        download_file_from_google_drive(faiss_pkl_id, faiss_pkl_path)
        
        # Dateien validieren
        if validate_files(temp_dir):
            st.success("Dateien erfolgreich heruntergeladen und validiert.")
            
            # HuggingFace Embeddings initialisieren
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # FAISS-Datenbank laden
            st.text("Loading FAISS vector store...")
            vectorstore = FAISS.load_local(temp_dir, embeddings, "index",allow_dangerous_deserialization=True)
            st.success("FAISS-Datenbank erfolgreich geladen!")
        else:
            st.error("Fehler bei der Dateivalidierung. Bitte überprüfen Sie die Google Drive-IDs.")
            st.stop()
    else:
        vectorstore = load_faiss_from_drive()
        
except Exception as e:
    st.error(f"Fehler beim Laden der FAISS-Datenbank: {e}")
    st.stop()

# Groq LLM initialisieren
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-70b-8192",  # Kann angepasst werden, je nach Präferenz
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
