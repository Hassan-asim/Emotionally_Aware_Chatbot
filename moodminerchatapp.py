#  Gemini Chat GUI with Local Emotion Detection
# Interact with the chatbot below.

import os
import joblib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

def preprocess(text, nlp):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-' and token.lemma_.strip() != '']
    return ' '.join(tokens)

# Load API key from environment (safe for GitHub use)
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

# Load the model and vectorizer with explicit numpy array handling
try:
    model = joblib.load('model.pkl', mmap_mode=None)
    vectorizer = joblib.load('vectorizer.pkl', mmap_mode=None)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Load spaCy model
import spacy
import spacy.cli

@st.cache_resource
def load_spacy_model():
    spacy.cli.download("en_core_web_sm")
    return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Streamlit UI
st.set_page_config(layout="wide")

# --- Theme and Logo ---
st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"], key="theme_selectbox")
if os.path.exists("logo.png"):
    logo = Image.open("logo.png")
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning("logo.png not found!")

if st.sidebar.button("New Chat", key="new_chat_button"):
    st.session_state['chat_history'] = []

# --- Custom CSS for modern UI ---
light_theme = """
    --primary-bg-color: #FFF4E6;
    --secondary-bg-color: #FCEBD9;
    --text-color: #1F2937;
    --secondary-text-color: #6B7280;
    --user-msg-bg: linear-gradient(90deg, #FFB347 0%, #FF7F50 100%);
    --bot-msg-bg: #EAEAEA;
    --bot-msg-text: #1F2937;
    --accent-color: #FF8C42;
    --accent-color-hover: #FFA552;
    --border-color: #D1D5DB;
    --input-bg-color: #F9FAFB;
"""
dark_theme = """
    --primary-bg-color: #121212;
    --secondary-bg-color: #1E1E1E;
    --text-color: #F3F4F6;
    --secondary-text-color: #9CA3AF;
    --user-msg-bg: linear-gradient(90deg, #FFB347 0%, #FF7F50 100%);
    --bot-msg-bg: #3A3A3A;
    --bot-msg-text: #F3F4F6;
    --accent-color: #FF9E4F;
    --accent-color-hover: #FFA86A;
    --border-color: #2C2C2C;
    --input-bg-color: #1E1E1E;
"""

st.markdown(f"""
    <style>
        :root {{
            {(dark_theme if theme == "Dark" else light_theme)}
        }}
        .stApp {{
            background: var(--primary-bg-color);
            color: var(--text-color);
            transition: all 0.3s ease-in-out;
        }}
        .stApp::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAbwAAAG8B8aLc+gAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAFpSURBVFiF7ZY/S8NQFMZ/Z+8sS1sIIiIiOKgoiGhB0J9QcXFx8B/oK+jS1aV/QRAcBJdCFFzU6tq1a/8xEUH8k/yQcE/eJz3J+5Jz7v3ec+85Vwgh8M8GAgEBAgQIECCQSiBwAc48sAUYgAE4gAN4gAecwAmswAqsQAuswAbsQA/cwA3cwA28wA+8wA/8wA8cQBEcQREcQhE8QhM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxM8QxMhAAAABJRU5ErkJggg==);
            opacity: 0.05;
            transform: rotate(15deg);
            z-index: 0;
            pointer-events: none;
        }}
        .stSidebar .stButton button {{
            background-color: var(--accent-color);
            color: white;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease-in-out;
        }}
        .stSidebar .stButton button:hover {{
            background-color: var(--accent-color-hover);
        }}
        .stSidebar {{
            background-color: var(--secondary-bg-color);
            box-shadow: inset -1px 0 0 #E5E7EB;
            padding: 1.25rem;
            transition: all 0.3s ease-in-out;
        }}
        .stSidebar h1, .stSidebar .stSelectbox label {{
            color: var(--text-color) !important;
        }}
        .stSidebar img {{
            transition: all 0.3s ease-in-out;
        }}
        .stSidebar img:hover {{
            transform: scale(1.05);
            filter: brightness(1.1);
        }}
        .stTextInput, .stTextArea {{
            background-color: var(--input-bg-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            color: var(--text-color);
            transition: all 0.3s ease-in-out;
        }}
        .stTextInput:focus-within, .stTextArea:focus-within {{
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px var(--accent-color-hover);
        }}
        .stTextInput ::placeholder, .stTextArea ::placeholder {{
            color: #9CA3AF;
            transition: all 0.3s ease-in-out;
        }}
        .stTextInput:focus-within ::placeholder, .stTextArea:focus-within ::placeholder {{
            opacity: 0;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            padding: 1.25rem;
            overflow-y: auto;
            flex-grow: 1;
        }}
        .chat-message {{
            padding: 15px;
            border-radius: 15px;
            margin: 10px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            border: 1px solid var(--border-color);
        }}
        .user-message-container {{
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }}
        .bot-message-container {{
            display: flex;
            justify-content: flex-start;
            width: 100%;
        }}
        .user-message {{
            background: var(--user-msg-bg);
            color: white;
            align-self: flex-end;
        }}
        .bot-message {{
            background-color: var(--bot-msg-bg);
            color: var(--bot-msg-text);
            align-self: flex-start;
        }}
        @media (max-width: 768px) {{
            .chat-message {{
                max-width: 90%;
            }}
        }}
    </style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists("logo.png"):
        st.image(logo, width=192)
with col2:
    st.title('Mood Miner – Your Emotional Chat Assistant')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Chat History Display ---
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state['chat_history']:
        st.markdown(f'<div class="user-message-container"><div class="chat-message user-message"><b>You:</b> {msg["user"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message-container"><div class="chat-message bot-message"><b>Pixie (Emotion: *{msg["emotion"]}*):</b> {msg["bot"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- User Input ---
if user_input := st.chat_input("Say something"):
    cleaned = preprocess(user_input, nlp)
    vect = vectorizer.transform([cleaned])
    probs = model.predict_proba(vect)[0]
    max_prob = np.max(probs)
    emotion = model.classes_[np.argmax(probs)]
    threshold = 0.5

    if max_prob < threshold:
        emotion = 'neutral'
        prompt = (
            f"You are bob, a friendly and intelligent assistant. The user said: '{user_input}'. "
            f"Your goal is to have a normal, helpful conversation. "
            f"Respond to the user's query directly and naturally. "
            f"Maintain a friendly and engaging tone."
        )
    else:
        prompt = (
            f"You are bob, an empathetic and intelligent assistant. The. The user is feeling '{emotion}' and said: '{user_input}'. "
            f"Your goal is to provide a supportive and insightful response. "
            f"First, respond directly to the user's query. "
            f"Then, at the end of your response, provide a relevant suggestion, quote, song, or activity to help them based on their emotion. "
            f"All suggestions should be generated by you, the LLM. "
            f"Avoid mentioning the detected emotion directly unless it feels natural and contextually appropriate. "
            f"Maintain a friendly and caring tone."
        )

    if st.session_state['chat_history']:
        history = '\n'.join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in st.session_state['chat_history']])
        prompt = f"Previous conversation:\n{history}\n\n" + prompt

    chat = genai.GenerativeModel('gemini-1.5-flash').start_chat()
    gemini_response = chat.send_message(prompt).text
    st.session_state['chat_history'].append({'user': user_input, 'bot': gemini_response, 'emotion': emotion})
    st.rerun()