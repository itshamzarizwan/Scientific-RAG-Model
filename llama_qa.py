import json
import requests
import streamlit as st

def query_llama3_ollama(question, retrieved_docs, model_name="llama3"):
    retrieved_context = ""
    references = {}

    for chunk in retrieved_docs:
        paper_id = chunk["paper_id"]
        text = chunk["text"]
        retrieved_context += f"[arXiv:{paper_id}] {text}\n\n"
        references[paper_id] = True

    prompt = f"""You are a helpful scientific research assistant. Based on the following sources from arXiv, answer the user's question below. Include inline citations like [arXiv:ID].

Sources:
{retrieved_context}

Question: {question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    raw_answer = ""
    placeholder = st.chat_message("ai").empty()
    for line in response.iter_lines(decode_unicode=True):
        if line:
            part = json.loads(line)["response"]
            raw_answer += part
            placeholder.markdown(raw_answer + "▌")

    # Process citation IDs
    cited_ids = [pid for pid in references if f"[arXiv:{pid}]" in raw_answer]
    id_to_num = {pid: str(i + 1) for i, pid in enumerate(cited_ids)}

    # Replace inline citations
    processed_answer = raw_answer
    for pid, num in id_to_num.items():
        processed_answer = processed_answer.replace(f"[arXiv:{pid}]", f"[{num}]")

    # ✅ Final rendering (correctly unindented)
    full_response = (
        processed_answer +
        "\n\n---\n\n" +
        "**📚 References:**\n\n" +
        "\n".join([f"[{num}] arXiv:{pid}" for pid, num in id_to_num.items()])
    )
    placeholder.markdown(full_response)

    # Return values for history/log
    ref_text = "References:\n"
    for pid, num in id_to_num.items():
        ref_text += f"[{num}] arXiv:{pid}\n"

    return full_response, ref_text.strip()

