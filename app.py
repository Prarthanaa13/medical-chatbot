import streamlit as st
import requests

# Hugging Face Model (Medical QA)
API_URL = "https://api-inference.huggingface.co/models/GonzaloValdenebro/MedicalQuestionAnswering"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

# Function to query Hugging Face model
def ask_medical_bot(question, history):
    # Add context to make answers more relevant
    context = " ".join([f"User: {u}\nBot: {b}" for u, b in history])
    prompt = f"User asked: {question}\n\nMedical assistant should provide clear, safe, and helpful medical information.\n\nChat history:\n{context}"
    
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()
    return data.get("answer") or data.get("generated_text") or "I'm not sure, could you please rephrase?"

# Streamlit UI
st.set_page_config(page_title="🩺 Medical Chatbot", page_icon="💊")
st.title("🩺 AI Medical Assistant")
st.write("Ask me any medical question. I will try to provide safe and relevant answers.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# User input
user_input = st.chat_input("Type your medical question here...")

if user_input:
    # Show user message
    st.chat_message("user").write(user_input)
    st.session_state.history.append(("user", user_input))

    # Get bot response
    with st.chat_message("assistant"):
        bot_response = ask_medical_bot(user_input, st.session_state.history)
        st.write(bot_response)
    st.session_state.history.append(("assistant", bot_response))

# Button to clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    st.experimental_rerun()
