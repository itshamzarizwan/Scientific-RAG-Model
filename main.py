import streamlit as st
import json
from data_loader import load_index_and_chunks
from retriever import retrieve_top_k
from llama_qa import query_llama3_ollama

# Streamlit app config
st.set_page_config(page_title="🧠 ArXiv Research Assistant", layout="wide")
st.title("🧠 ArXiv RAG Research Assistant")

# Load FAISS index, chunks, and model (cached)
@st.cache_resource
def init_resources():
    return load_index_and_chunks()

index, chunks, embedding_model = init_resources()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("human"):
        st.markdown(msg["user"])
    with st.chat_message("ai"):
        st.markdown(msg["bot"])

# Chat input at bottom
if query := st.chat_input("Type your scientific question here..."):
    with st.chat_message("human"):
        st.markdown(query)

    with st.spinner("Thinking with LLaMA3..."):
        top_chunks = retrieve_top_k(query, index, chunks, embedding_model)
        answer, refs = query_llama3_ollama(query, top_chunks)  # This already streams!

    # Save answer for next rerun
    st.session_state.messages.append({"user": query, "bot": answer})

