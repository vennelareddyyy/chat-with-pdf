import streamlit as st
from rag_pipeline import create_vector_store
from langchain_ollama import OllamaLLM
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile

st.set_page_config(page_title="AI Document Assistant", page_icon="🤖")

st.title("🤖 AI Document Assistant")
st.caption("Chat with your PDFs using RAG")

# Sidebar
st.sidebar.title("📂 Uploaded Documents")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    file_paths = []

    for file in uploaded_files:

        st.sidebar.write(file.name)

        with open(file.name, "wb") as f:
            f.write(file.read())

        file_paths.append(file.name)

    vectorstore = create_vector_store(file_paths)

    llm = OllamaLLM(model="gemma:2b")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Voice input
    voice = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop")

    user_input = None

    if voice and "bytes" in voice:

        audio_bytes = voice["bytes"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name

        model = WhisperModel("base")

        segments, info = model.transcribe(audio_path)

        text = ""

        for segment in segments:
            text += segment.text

        user_input = text

    # Text input
    if not user_input:
        user_input = st.chat_input("Ask something about your PDFs")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        docs = vectorstore.similarity_search(user_input, k=3)

        context = "\n".join([doc.page_content for doc in docs])

        history = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
        )

        prompt = f"""
You are an AI assistant answering questions about documents.

Conversation history:
{history}

Context:
{context}

Question:
{user_input}

Give a short answer using the context.
"""

        response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        st.subheader("📄 Sources")

        for doc in docs:
            st.info(
                f"Page {doc.metadata.get('page', 'unknown')}:\n\n{doc.page_content[:300]}"
            )