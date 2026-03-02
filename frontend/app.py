import streamlit as st
import requests
import uuid

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Academic QA", layout="centered")

st.title("📚 ScholarRAG QA Assistant")
st.markdown("### Retrieval-Augmented Academic QA System")

# Generate session id once
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = requests.post(
                API_URL,
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id
                }
            )

            data = response.json()

            answer = data["answer"]
            sources = data.get("sources", [])

            formatted = answer

            if sources:
                formatted += "\n\n**Sources:**\n"
                for s in sources:
                    formatted += f"- {s}\n"

            st.markdown(formatted)

    st.session_state.messages.append(
        {"role": "assistant", "content": formatted}
    )
st.markdown("---")
st.caption("Built with ❤️ using FastAPI, Streamlit, ChromaDB & Groq LLaMA")
