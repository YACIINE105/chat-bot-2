import streamlit as st
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Tourism Chatbot EG",
    page_icon="🌍",
    layout="wide"
)

# Title with better styling
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        Tourism Assistant
    </h1>
""", unsafe_allow_html=True)

# Load Q&A data from q.txt file
QA_DATA = []
try:
    if os.path.exists('q.txt'):
        with open('q.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            # Split by numbered entries like "1. Q:" and process each one
            qa_items = content.split('\n\n')
            for item in qa_items:
                if 'Q:' in item and 'A:' in item:
                    question_part = item.split('A:')[0].split('Q:')[1].strip()
                    answer_part = item.split('A:')[1].strip()
                    QA_DATA.append({
                        "question": question_part,
                        "answer": answer_part
                    })
        st.sidebar.success(f"Loaded {len(QA_DATA)} Q&A pairs from q.txt")
    else:
        st.sidebar.warning("q.txt file not found. Proceeding without Q&A data.")
except Exception as e:
    st.sidebar.error(f"Error loading Q&A data: {str(e)}")

# Create .env file if it doesn't exist
if not os.path.exists('.env'):
    with open('.env', 'w') as f:
        f.write('GEMINI_API_KEY=AIzaSyCZ-1D3PVny0WD2Qzz_2rYi-C2a7VWv24Q\n')

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_request" not in st.session_state:
    st.session_state.last_request = datetime.now() - timedelta(seconds=10)

if "cache" not in st.session_state:
    st.session_state.cache = {}

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Get API key from .env file or environment variables
api_key = os.getenv('GEMINI_API_KEY')

# Set up configuration settings in sidebar
st.sidebar.header("Configuration")

# API key management (moved to top of sidebar for better visibility)
st.sidebar.subheader("API Key Management")
new_api_key = st.sidebar.text_input("Gemini API Key", value=api_key, type="password")

if st.sidebar.button("Update API Key"):
    with open('.env', 'w') as f:
        f.write(f'GEMINI_API_KEY={new_api_key}\n')
    api_key = new_api_key
    st.sidebar.success("API key updated!")

gemini_model = st.sidebar.selectbox(
    "Select Gemini Model",
    ["gemini-2.0-flash-lite", "gemini-2.0-flash"],
    index=0
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Higher values make output more random, lower values make it more deterministic"
)

# Function to find matching Q&A
def find_qa_match(query, threshold=0.6):
    """Find the best matching Q&A for the user query"""
    if not query or not QA_DATA:
        return None, 0
    
    query = query.lower().strip()
    best_match = None
    best_score = 0
    
    for qa_pair in QA_DATA:
        question = qa_pair["question"].lower()
        
        # Simple word match scoring
        query_words = set(query.split())
        question_words = set(question.split())
        common_words = query_words.intersection(question_words)
        
        if common_words:
            # Calculate Jaccard similarity: intersection over union
            score = len(common_words) / (len(query_words.union(question_words)))
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = qa_pair["answer"]
    
    return best_match, best_score

def call_gemini_api(prompt, history=None):
    """
    Call the Gemini API with the given prompt and history
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Format conversation history
    contents = []
    
    # Add history if provided
    if history:
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
    
    # Add current prompt
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    data = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.8,
            "topK": 40
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        
        # Extract the response text
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0]:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    return content["parts"][0]["text"]
        
        # If we couldn't extract the text through the expected path
        return "I'm sorry, I couldn't generate a response."
    
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."

# Function to prepare Q&A context for prompts
def prepare_qa_context():
    """Create a formatted context string from Q&A data"""
    if not QA_DATA:
        return ""
    
    context = "Here are some common questions and answers about Egypt tourism:\n\n"
    for qa in QA_DATA[:15]:  # Limit to first 15 to avoid token limits
        context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
    
    return context

# Display chat history with improved formatting
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🌍"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Enhanced rate limiting
if (datetime.now() - st.session_state.last_request).seconds < 1:
    st.warning("Please wait a moment before sending another message.")
    st.stop()

# User input
prompt = st.chat_input("Ask about destinations, culture, or safety tips...")

if prompt:
    st.session_state.last_request = datetime.now()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner("Generating response..."):
            try:
                # Check cache first
                cache_key = prompt.lower()
                if cache_key in st.session_state.cache:
                    response = st.session_state.cache[cache_key]
                else:
                    # First check Q&A data for direct matches
                    qa_response, qa_score = find_qa_match(prompt)
                    
                    if qa_response and qa_score > 0.7:  # Higher threshold for Q&A
                        response = qa_response
                    else:
                        # Prepare context from Q&A
                        qa_context = prepare_qa_context()
                        
                        # Use the Gemini API with Q&A context
                        full_prompt = f"""You are a tourism assistant specializing in Egypt travel.
                        
                        {qa_context}
                        
                        Question: {prompt}

                        Answer the question about Egypt tourism. Consider:
                        - Historical sites and attractions
                        - Cultural information
                        - Practical travel advice
                        - Safety information
                        - Local customs

                        Answer concisely but informatively, referencing the provided Q&A context above when applicable."""

                        # Get response from Gemini API, passing conversation history
                        response = call_gemini_api(full_prompt, st.session_state.conversation_history[-5:] if len(st.session_state.conversation_history) > 0 else None)

                    # Store in cache
                    st.session_state.cache[cache_key] = response

                # Display cleaned response
                cleaned_response = response.strip()
                st.markdown(cleaned_response)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
                st.session_state.conversation_history.append({"role": "assistant", "content": cleaned_response})

            except Exception as e:
                error_response = f"Sorry, I encountered an error: {str(e)}. Please try again later."
                st.error(error_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_response
                })

# Sidebar with deployment-ready information
with st.sidebar:
    st.header("ℹ️ About")

    st.markdown(f"""
    **Tourism Expert Chatbot**  
    • Powered by Google's {gemini_model}
    • Provides detailed travel information  
    • Offers cultural insights  
    • Uses Q&A data from q.txt file
    • Remembers conversation history  
    • Fast responses for common questions
    """)

    # Add clear cache button
    if st.button("Clear Response Cache"):
        st.session_state.cache = {}
        st.success("Cache cleared!")

    # Add reset button to clear conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.experimental_rerun()
        
    # Q&A Data Status
    st.subheader("Q&A Data")
    if QA_DATA:
        st.success(f"Loaded {len(QA_DATA)} Q&A items")
    else:
        st.warning("No Q&A data loaded. Place q.txt in the same directory.")
        
    if st.button("Reload Q&A Data"):
        st.experimental_rerun()
