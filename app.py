import streamlit as st
import requests
import json # Import json to pretty-print error responses

# Hugging Face Model (Medical QA)
# Ensure this API_URL is correct for the specific model
API_URL = "https://api-inference.huggingface.co/models/GonzaloValdenebro/MedicalQuestionAnswering"

# Headers for the API call, including the authorization token
# This block attempts to get the API key, and stops the app if it's missing.
try:
    headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}
except KeyError:
    st.error("Hugging Face API Key (HF_API_KEY) not found in Streamlit secrets. Please add it via 'Manage app' -> 'Secrets'.")
    st.stop() # Stop the app if the key is missing to prevent further errors

# Function to query Hugging Face model
def ask_medical_bot(question, history):
    # Construct a detailed prompt for the medical assistant,
    # including the current question and previous chat history for context.
    # The context is formatted to show User and Bot turns clearly.
    context_lines = []
    for role, msg in history:
        if role == "user":
            context_lines.append(f"User: {msg}")
        else:
            context_lines.append(f"Assistant: {msg}")
    
    full_context = "\n".join(context_lines)

    # The prompt guides the AI to act as a helpful medical assistant.
    # It emphasizes providing safe, relevant, and clear information.
    # The current question is framed as a user query, followed by chat history.
    prompt = (
        f"You are an AI Medical Assistant. Your goal is to provide clear, safe, and helpful medical information "
        f"based on the user's questions and the provided chat history. Do not give medical advice that requires a diagnosis "
        f"or treatment, and always advise consulting a healthcare professional for serious conditions. "
        f"Be concise and focus on common knowledge. If you cannot provide a safe or relevant answer, politely decline.\n\n"
        f"Current User Question: {question}\n\n"
        f"Chat History:\n{full_context}\n\n"
        f"Assistant's Response:"
    )
    
    # Prepare the payload for the API request.
    # The 'inputs' key contains the constructed prompt.
    payload = {"inputs": prompt}

    try:
        # Make the POST request to the Hugging Face Inference API.
        # Added a timeout of 60 seconds for robustness in case the model is slow.
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        # Streamlit provides a way to show internal details for debugging without crashing the app.
        # These DEBUG messages will appear in your Streamlit Cloud logs.
        st.write(f"DEBUG: API Response Status Code: {response.status_code}")
        st.write(f"DEBUG: Raw API Response: {json.dumps(data, indent=2)}")

        # Check for common keys in the response from Hugging Face models
        # Some models return 'generated_text', others 'answer' (for QA models).
        if isinstance(data, list) and len(data) > 0:
            if "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif "answer" in data[0]: # For QA models that might return a list of answers
                return data[0]["answer"]
        elif isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            elif "answer" in data:
                return data["answer"]
            elif "error" in data: # Explicitly check for an 'error' key in the response
                st.error(f"Hugging Face API Error: {data['error']}")
                return f"An error occurred with the AI model: {data['error']}. Please try again."

        # Fallback if no expected key is found in a successful API response
        return "I'm not sure, could you please rephrase or ask a different question?"

    except requests.exceptions.HTTPError as http_err:
        # Log and display HTTP errors
        st.error(f"HTTP error occurred: {http_err} - Response text: {response.text}")
        return f"An HTTP error occurred: {http_err}. Details: {response.text}. Please try again later."
    except requests.exceptions.ConnectionError as conn_err:
        # Log and display connection errors
        st.error(f"Connection error occurred: {conn_err}")
        return "A connection error occurred. The AI service might be unavailable. Please check your internet connection or try again later."
    except requests.exceptions.Timeout as timeout_err:
        # Log and display timeout errors
        st.error(f"Timeout error occurred: {timeout_err}")
        return "The AI model took too long to respond. Please try again or simplify your question."
    except requests.exceptions.RequestException as req_err:
        # Log and display any other request-related errors
        st.error(f"An unexpected request error occurred: {req_err}")
        return "An unexpected error occurred while communicating with the AI. Please try again."
    except json.JSONDecodeError as json_err:
        # Log and display JSON decoding errors (if the response isn't valid JSON)
        st.error(f"JSON decode error: {json_err} - Response text: {response.text}")
        return "The AI model returned an unreadable response. Please try again."
    except Exception as e:
        # Catch any other unforeseen errors
        st.error(f"An unforeseen error occurred: {e}")
        return f"An internal error occurred: {e}. Please try again."

# Streamlit UI configuration
st.set_page_config(page_title="🩺 Medical Chatbot", page_icon="💊")
st.title("🩺 AI Medical Assistant")
st.write("Ask me any medical question. I will try to provide safe and relevant answers.")

# Initialize session state for chat history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Display all messages in the chat history
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

# User input section
user_input = st.chat_input("Type your medical question here...")

if user_input:
    # Add user message to history and display it in the chat
    st.chat_message("user").write(user_input)
    st.session_state.history.append(("user", user_input))

    # Get bot response and display it, showing a spinner while waiting
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Visual feedback that the bot is processing
            bot_response = ask_medical_bot(user_input, st.session_state.history)
            st.write(bot_response)
    
    # Add bot's response to the chat history
    st.session_state.history.append(("assistant", bot_response))

# Button to clear chat history
if st.button("🗑️ Clear Chat"):
    st.session_state.history = []
    # Use st.rerun() instead of st.experimental_rerun() for current Streamlit versions
    st.rerun()
