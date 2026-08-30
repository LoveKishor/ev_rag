import os
import uuid
import tempfile
import hashlib  # still used for feedback button keys

import streamlit as st
from dotenv import load_dotenv

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EV Assistant",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# 2. IMPORTS
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as Qdrant   # correct import
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ============================================================
# 3. CONFIGURATION
# ============================================================
load_dotenv()

# Qdrant Cloud settings
QDRANT_URL = os.getenv("QDRANT_URL")          # e.g. "https://xyz-cluster.cloud.qdrant.io"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ev_docs")
MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "ev_memory")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
RERANK_K = int(os.getenv("RERANK_K", "4"))
MEMORY_K = int(os.getenv("MEMORY_K", "2"))
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"

# ============================================================
# 4. SECRETS / KEY HELPERS
# ============================================================
def get_secret_or_env(name: str) -> str:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return str(os.getenv(name, "")).strip()

def sanitize_api_key(key: str | None) -> str:
    if not key:
        return ""
    return str(key).strip().encode("ascii", "ignore").decode("ascii")

def get_admin_groq_key() -> str:
    return sanitize_api_key(get_secret_or_env("GROQ_API_KEY"))

# ============================================================
# 5. SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "groq_api_key": "",
    "user_provided_groq_key": "",
    "upload_authorized": False,
    "feedback_authorized": False,
    "feedback_enabled": False,
    "messages": [],
    "waiting_for_correction": False,
    "correction_prompt": "",
    "selected_model": GROQ_MODEL,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# 6. CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner="Loading reranker...")
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def get_qdrant_client():
    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("QDRANT_URL and QDRANT_API_KEY must be set in secrets/.env")
        st.stop()
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_resource
def get_vector_store(collection: str):
    """
    Return a Qdrant vector store for the given collection.
    The collection is created if it doesn't exist.
    """
    client = get_qdrant_client()
    # Check if collection exists; if not, create it
    collections = client.get_collections().collections
    if not any(c.name == collection for c in collections):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    return Qdrant(
        client=client,
        collection_name=collection,
        embedding=get_embeddings(),   # ✅ singular
    )

@st.cache_resource
def get_available_models(api_key: str):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        st.error(f"Could not fetch models: {e}")
        return ["mixtral-8x7b-32768", "gemma2-9b-it", "llama3-70b-8192"]

@st.cache_resource
def get_llm(api_key: str):
    safe_key = sanitize_api_key(api_key)
    if not safe_key:
        raise ValueError("Groq API key is empty.")
    model = st.session_state.get("selected_model", GROQ_MODEL)
    return ChatGroq(
        model=model,
        temperature=0.2,
        max_tokens=1200,
        groq_api_key=safe_key,
    )

@st.cache_resource
def get_final_prompt():
    return PromptTemplate.from_template(
        """
You are an expert EV diagnostic assistant.

Answer ONLY from the supplied document context and relevant past lessons.
Do not invent specifications, fault codes, measurements, procedures, or causes.

If the supplied context does not contain enough information, say:
"Insufficient information in the knowledge base."

For diagnostic questions:
1. State the most likely diagnosis or interpretation.
2. Give the supporting evidence from the context.
3. Give practical next checks when the context supports them.
4. Clearly distinguish facts from recommendations.

Past lessons:
{memory_text}

Document context:
{context}

Question:
{question}

Answer:
""".strip()
    )

# ============================================================
# 7. PDF INGESTION (using UUIDs as IDs)
# ============================================================
def ingest_pdfs(uploaded_files, collection: str = COLLECTION_NAME) -> int:
    if not uploaded_files:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for uploaded_file in uploaded_files:
        temp_path = None
        try:
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            docs = PyPDFLoader(temp_path).load()
            chunks = splitter.split_documents(docs)

            for chunk_index, chunk in enumerate(chunks):
                page = chunk.metadata.get("page", 0)
                text = chunk.page_content.strip()
                if not text:
                    continue
                # ✅ Use UUID for Qdrant compatibility
                doc_id = str(uuid.uuid4())
                all_chunks.append(text)
                all_ids.append(doc_id)
                all_metadatas.append({
                    "source": uploaded_file.name,
                    "page": int(page) + 1,
                    "chunk": chunk_index,
                })
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        return 0

    vector_store = get_vector_store(collection)
    vector_store.add_texts(
        texts=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
        batch_size=100,
    )
    return len(all_chunks)

# ============================================================
# 8. RETRIEVAL
# ============================================================
def direct_retrieve(question: str, k: int = RETRIEVE_K):
    if not question.strip():
        return []
    vector_store = get_vector_store(COLLECTION_NAME)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def hyde_retrieve(question: str, k: int = RETRIEVE_K):
    if not USE_HYDE:
        return direct_retrieve(question, k)

    try:
        llm = get_llm(st.session_state.groq_api_key)
        prompt = (
            "Write a short factual passage containing the concepts "
            "likely to appear in a technical EV document that answers "
            "this question. Do not invent specific values.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        hypothetical = llm.invoke(prompt).content
        vector_store = get_vector_store(COLLECTION_NAME)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(hypothetical)
    except Exception:
        return direct_retrieve(question, k)

def rerank_documents(query: str, documents, top_k: int = RERANK_K):
    if not documents:
        return []
    if len(documents) <= top_k:
        return documents

    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs, batch_size=8, show_progress_bar=False)
    ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# ============================================================
# 9. MEMORY
# ============================================================
def retrieve_memory(question: str, k: int = MEMORY_K):
    vector_store = get_vector_store(MEMORY_COLLECTION)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def save_to_memory(question: str, answer: str):
    text = f"Question: {question}\nAnswer: {answer}"
    doc_id = str(uuid.uuid4())   # ✅ UUID
    vector_store = get_vector_store(MEMORY_COLLECTION)
    vector_store.add_texts(
        texts=[text],
        metadatas=[{"type": "user_feedback"}],
        ids=[doc_id],
    )

# ============================================================
# 10. ANSWER GENERATION
# ============================================================
def generate_final_answer(question: str, context_docs, memory_docs):
    if not context_docs:
        return "I could not find relevant information in the EV knowledge base for this question."

    llm = get_llm(st.session_state.groq_api_key)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    memory_text = "\n\n".join(doc.page_content for doc in memory_docs) if memory_docs else "None"

    prompt = get_final_prompt()
    response = llm.invoke(
        prompt.format(
            memory_text=memory_text,
            context=context_text,
            question=question,
        )
    )
    return response.content

# ============================================================
# 11. GROQ ERROR HANDLING
# ============================================================
def explain_groq_error(error: Exception) -> str:
    message = str(error)
    if "401" in message or "invalid_api_key" in message.lower():
        return (
            "❌ Groq authentication failed. The API key is invalid, "
            "expired, revoked, or not the key you intended to use. "
            "Create a new Groq key and replace GROQ_API_KEY, or enter "
            "a valid user key."
        )
    if "model_not_found" in message.lower() or "unavailable" in message.lower():
        return (
            f"❌ The selected model is not available for your key. "
            f"Please choose another model from the dropdown in the sidebar."
        )
    if "decommissioned" in message.lower():
        return (
            f"❌ Groq model '{st.session_state.get('selected_model', GROQ_MODEL)}' has been decommissioned. "
            "Select a different model from the sidebar."
        )
    return f"❌ Groq request failed: {message}"

# ============================================================
# 12. SIDEBAR (unchanged)
# ============================================================
st.title("🧠 EV Assistant")

with st.sidebar:
    st.header("🤖 Groq API")

    is_admin = st.session_state.upload_authorized or st.session_state.feedback_authorized

    if is_admin:
        admin_key = get_admin_groq_key()
        if admin_key:
            st.session_state.groq_api_key = admin_key
            st.success("✅ Admin Groq key loaded.")
        else:
            st.error("GROQ_API_KEY is not configured in Streamlit Secrets/.env.")
        if st.button("🚪 Logout (exit admin mode)"):
            st.session_state.upload_authorized = False
            st.session_state.feedback_authorized = False
            st.session_state.groq_api_key = ""
            st.rerun()
    else:
        st.info("🔑 Enter your own Groq API key to chat")
        user_key = st.text_input(
            "Your Groq API key",
            type="password",
            key="groq_user_input",
        )
        if user_key:
            cleaned = sanitize_api_key(user_key)
            if cleaned:
                st.session_state.user_provided_groq_key = cleaned
                st.session_state.groq_api_key = cleaned
                st.success("✅ Groq key accepted.")

    if st.session_state.groq_api_key:
        available = get_available_models(st.session_state.groq_api_key)
        if available:
            current = st.session_state.get("selected_model", available[0])
            if current not in available:
                current = available[0]
            selected = st.selectbox(
                "Select Groq model",
                options=available,
                index=available.index(current),
                key="model_selector"
            )
            if selected != st.session_state.get("selected_model"):
                st.session_state.selected_model = selected
                st.cache_resource.clear()
                st.rerun()
            st.caption(f"Using: `{st.session_state.selected_model}`")
        else:
            st.warning("No models available. Check your API key.")

    st.divider()
    st.header("🔒 Upload Protection")

    upload_password = st.text_input(
        "Upload password",
        type="password",
        key="upload_password_input",
    )

    if st.button("🔓 Verify Upload Password"):
        correct_password = get_secret_or_env("UPLOAD_PASSWORD")
        if correct_password and upload_password == correct_password:
            st.session_state.upload_authorized = True
            st.success("✅ Upload access granted.")
            admin_key = get_admin_groq_key()
            if admin_key:
                st.session_state.groq_api_key = admin_key
            st.rerun()
        else:
            st.session_state.upload_authorized = False
            st.error("❌ Wrong password.")

    st.divider()
    st.header("👍 Feedback Mode")

    feedback_password = st.text_input(
        "Admin password",
        type="password",
        key="feedback_password_input",
    )

    if st.button("🔑 Authorize Feedback"):
        correct_password = get_secret_or_env("UPLOAD_PASSWORD")
        if correct_password and feedback_password == correct_password:
            st.session_state.feedback_authorized = True
            admin_key = get_admin_groq_key()
            if admin_key:
                st.session_state.groq_api_key = admin_key
            st.success("✅ Feedback settings unlocked.")
            st.rerun()
        else:
            st.session_state.feedback_authorized = False
            st.error("❌ Wrong password.")

    feedback_toggle = st.checkbox(
        "Allow User Feedback (👍/👎)",
        value=st.session_state.feedback_enabled,
        disabled=not st.session_state.feedback_authorized,
        key="feedback_checkbox",
    )
    if st.session_state.feedback_authorized:
        st.session_state.feedback_enabled = feedback_toggle

    st.divider()

    if st.session_state.upload_authorized:
        st.header("📤 Update Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("🚀 Add to Vector DB"):
            with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                try:
                    count = ingest_pdfs(uploaded_files)
                    if count:
                        st.success(f"✅ Added {count} chunks to Qdrant.")
                    else:
                        st.warning("No readable text chunks were found.")
                except Exception as error:
                    st.error(f"❌ Upload failed: {error}")
    else:
        st.info("🔒 Upload area is locked.")

# ============================================================
# 13. STARTUP CHECKS
# ============================================================
try:
    get_qdrant_client()  # just verify connection
except Exception as e:
    st.error(f"❌ Qdrant connection error: {e}")
    st.stop()

if not st.session_state.groq_api_key:
    st.warning("⚠️ Enter a valid Groq API key in the sidebar to start chatting.")
    st.stop()

# ============================================================
# 14. CHAT HISTORY
# ============================================================
st.subheader("💬 Ask anything")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 15. CHAT PIPELINE
# ============================================================
if prompt := st.chat_input("Type your EV question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Searching EV knowledge base..."):
                retrieved_docs = hyde_retrieve(prompt, k=RETRIEVE_K)

            with st.spinner("🎯 Ranking relevant documents..."):
                top_docs = rerank_documents(prompt, retrieved_docs, top_k=RERANK_K)

            with st.spinner("🧠 Checking past lessons..."):
                memory_docs = retrieve_memory(prompt, k=MEMORY_K)

            with st.spinner("💬 Generating answer..."):
                answer = generate_final_answer(prompt, top_docs, memory_docs)

            st.markdown(answer)

        except Exception as error:
            answer = explain_groq_error(error)
            st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.session_state.feedback_enabled:
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Good", key=f"good_{hashlib.md5(prompt.encode()).hexdigest()}"):
                    try:
                        save_to_memory(prompt, answer)
                        st.toast("✅ Saved to memory.")
                    except Exception as error:
                        st.error(f"Memory save failed: {error}")
            with col2:
                if st.button("👎 Wrong", key=f"wrong_{hashlib.md5(prompt.encode()).hexdigest()}"):
                    st.session_state.correction_prompt = prompt
                    st.session_state.waiting_for_correction = True
                    st.rerun()

# ============================================================
# 16. CORRECTION WORKFLOW
# ============================================================
if st.session_state.waiting_for_correction:
    with st.chat_message("assistant"):
        st.warning("🤔 What is the correct answer?")
        correction = st.text_area("Correct answer", key="correction_input")
        if st.button("Submit Correction"):
            if correction.strip():
                try:
                    save_to_memory(st.session_state.correction_prompt, correction.strip())
                    st.success("✅ Correction saved for future similar questions.")
                    st.session_state.waiting_for_correction = False
                    st.session_state.correction_prompt = ""
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not save correction: {error}")
            else:
                st.error("Please enter a valid correction.")