import re
import os
import random
import numpy as np
import faiss
import requests
from functools import lru_cache
import urllib.parse
import math
from collections import Counter
import google.generativeai as genai
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="DecodeBot", page_icon="🤖", layout="wide")

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600&display=swap');
.stApp { background: linear-gradient(135deg, #FAF7F2 0%, #F5EDE8 50%, #FAF7F2 100%); font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 5rem; }
[data-testid="stSidebar"] { background: #FDF5F0 !important; border-right: 1px solid #EDD9D5 !important; }
[data-testid="stSidebar"] .stButton > button { background: transparent !important; color: #3D2B2B !important; border: 1px solid #D4A8A0 !important; border-radius: 20px !important; width: 100% !important; text-align: left !important; padding: 0.5rem 1rem !important; font-size: 0.85rem !important; margin-bottom: 0.3rem !important; font-weight: 400 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #F2E4E1 !important; border-color: #C9967E !important; }
.new-chat-btn > button { background: linear-gradient(135deg, #C9967E, #B07060) !important; color: white !important; border: none !important; border-radius: 20px !important; width: 100% !important; font-weight: 500 !important; margin-bottom: 1rem !important; }
.chat-header { text-align: center; padding: 1rem 0 0.3rem; }
.chat-header h1 { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #3D2B2B; margin: 0; }
.chat-header p { font-size: 0.78rem; color: #A08070; letter-spacing: 0.12em; text-transform: uppercase; margin: 0.25rem 0 0; }
.header-rule { height: 2px; background: linear-gradient(to right, transparent, #C9967E, transparent); margin: 0.6rem 4rem; border: none; }
.bubble-row { display: flex; align-items: flex-end; gap: 0.6rem; margin-bottom: 0.8rem; }
.bubble-row.user { flex-direction: row-reverse; }
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; flex-shrink: 0; }
.avatar.bot-av { background: #EDD9D5; border: 1.5px solid #C9967E; }
.avatar.user-av { background: #D4E8F0; border: 1.5px solid #7ABED4; }
.bubble { max-width: 68%; padding: 0.7rem 1rem; border-radius: 18px; font-size: 0.91rem; line-height: 1.55; }
.bubble.user { background: #C9967E; color: #fff; border-bottom-right-radius: 4px; }
.bubble.bot { background: #FFFFFF; color: #3D2B2B; border-bottom-left-radius: 4px; box-shadow: 0 1px 8px rgba(180,120,100,0.10); border: 1px solid #EDD9D5; }
.debug-badge { font-size: 0.67rem; color: #B08A7A; margin: 0.2rem 0 0.6rem 2.6rem; letter-spacing: 0.04em; opacity: 0.75; }
.welcome-box { text-align: center; padding: 2rem 1rem; color: #A08070; }
.welcome-box p { font-size: 0.9rem; line-height: 1.6; max-width: 380px; margin: 0.5rem auto 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; justify-content: center; }
.chip { background: #F2E4E1; border: 1px solid #D4A8A0; border-radius: 20px; padding: 0.3rem 0.85rem; font-size: 0.8rem; color: #7A4A3A; font-family: 'Inter', sans-serif; }
.stTextInput > div > div > input { background: #FFFFFF !important; border: 1.5px solid #D4A8A0 !important; border-radius: 24px !important; padding: 0.65rem 1.2rem !important; font-size: 0.92rem !important; color: #3D2B2B !important; }
.stTextInput > div > div > input:focus { border-color: #C9967E !important; box-shadow: 0 0 0 3px rgba(201,150,126,0.15) !important; }
.stTextInput > label { display: none !important; }
.stButton > button { background: linear-gradient(135deg, #C9967E, #B07060) !important; color: white !important; border: none !important; border-radius: 24px !important; padding: 0.6rem 1.4rem !important; font-size: 0.9rem !important; font-weight: 500 !important; width: 100% !important; }
.input-divider { height: 1px; background: linear-gradient(to right, transparent, #D4A8A0, transparent); margin: 0.8rem 0; }
[data-testid="stSidebar"] .stTextInput > div > div > input { border-radius: 20px !important; font-size: 0.82rem !important; padding: 0.4rem 0.9rem !important; border: 1px solid #D4A8A0 !important; background: #FFF8F5 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLEANING
# clean()         → for intent detection (strips symbols)
# clean_for_name  → for name handler only
# tools get RAW input so operators stay intact
# ============================================================

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())

# ============================================================
# INTENTS
# ============================================================

intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello! 👋 How can I help you today?", "Hi there! What's on your mind?", "Hey! Ask me anything. 🌸"]
    },
    "identity": {
        "patterns": ["who are you", "what are you", "your name", "introduce yourself"],
        "responses": ["I'm DecodeBot — a hybrid AI chatbot built with rules, semantic search, and Gemini LLM. 🤖"]
    },
    "ai": {
        "patterns": ["ai", "artificial intelligence", "what is ai", "tell me about ai"],
        "responses": ["AI (Artificial Intelligence) enables machines to simulate human intelligence — learning, reasoning, and problem solving. ✨"]
    },
    "python": {
        "patterns": ["python", "what is python", "tell me about python"],
        "responses": ["Python is a versatile, beginner-friendly language widely used in AI, data science, and web development. 🐍"]
    },
    "ml": {
        "patterns": ["machine learning", "ml", "what is machine learning"],
        "responses": ["Machine Learning is a subset of AI where systems learn patterns from data to make predictions. 📊"]
    },
    "help": {
        "patterns": ["help", "help me", "what can you do", "assist me"],
        "responses": ["I can answer questions about AI, Python, and Machine Learning. For anything else, Gemini steps in! 🚀"]
    },
    "farewell": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye! Come back anytime. 👋", "See you soon! 🌸"]
    }
}

# ============================================================
# EMBEDDER
# ============================================================

class SimpleEmbedder:
    def __init__(self): self.vocab = {}
    def build_vocab(self, texts):
        idx = 0
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
    def encode(self, texts):
        dim = max(len(self.vocab), 1)
        vectors = []
        for text in texts:
            vec = [0.0] * dim
            counts = Counter(text.split())
            for word, count in counts.items():
                if word in self.vocab:
                    vec[self.vocab[word]] += count
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return np.array(vectors, dtype=np.float32)

@st.cache_resource
def load_model_and_index():
    embedder = SimpleEmbedder()
    plist, imap = [], []
    for intent, data in intents.items():
        for p in data["patterns"]:
            plist.append(p)
            imap.append(intent)
    embedder.build_vocab(plist)
    embs = embedder.encode(plist)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs.astype(np.float32))
    return embedder, idx, imap

embedder, intent_index, intent_map = load_model_and_index()

# ============================================================
# GEMINI SETUP
# ⚠️  Put your API key here — never share this file publicly
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=API_KEY)

@st.cache_resource
def load_gemini():
    return genai.GenerativeModel("gemini-2.0-flash")   # ← updated model

gemini_model = load_gemini()

# ============================================================
# TOOLS  (receive RAW input — operators intact)
# ============================================================

def get_weather(raw):
    match = re.search(r'(?:weather\s+(?:in\s+)?|in\s+)([a-zA-Z]+)', raw, re.IGNORECASE)
    city = match.group(1) if match else "London"
    try:
        res = requests.get(f"https://wttr.in/{city}?format=%C+%t", timeout=5)
        if res.status_code == 200:
            return f"☁️ Weather in {city.title()}: {res.text.strip()}"
        return f"Couldn't fetch weather for {city}."
    except:
        return "Weather service unavailable right now."

def calculate(raw):
    """
    FIX: receives raw input so +, -, *, / are never stripped.
    Parses to float so 2+2=4 not "22".
    """
    text = raw.lower()
    for prefix in ["calculate", "what is", "solve", "evaluate", "compute"]:
        text = text.replace(prefix, "")
    text = text.strip()

    # match numbers + operator (floats supported too)
    match = re.search(r'([\d\.]+)\s*([\+\-\*\/])\s*([\d\.]+)', text)
    if not match:
        return None

    num1 = float(match.group(1))   # real float — no string concat
    op   = match.group(2)
    num2 = float(match.group(3))   # real float

    if   op == '+': result = num1 + num2
    elif op == '-': result = num1 - num2
    elif op == '*': result = num1 * num2
    elif op == '/':
        if num2 == 0: return "🧮 Cannot divide by zero!"
        result = num1 / num2
    else:
        return None

    def fmt(n): return int(n) if float(n).is_integer() else round(n, 4)
    return f"🧮 {fmt(num1)} {op} {fmt(num2)} = **{fmt(result)}**"

def search_web(raw):
    query = raw.lower()
    for p in ["search for", "search", "look up", "find", "google"]:
        query = query.replace(p, "")
    query = query.strip()
    if len(query) < 3:
        return "What should I search for?"
    try:
        encoded = urllib.parse.quote(query)
        r = requests.get(f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1", timeout=5)
        data = r.json()
        if data.get("Abstract"):
            return data["Abstract"]
        if data.get("RelatedTopics"):
            first = data["RelatedTopics"][0]
            if isinstance(first, dict) and first.get("Text"):
                return first["Text"]
        return f"🔍 [Search results](https://duckduckgo.com/?q={encoded})"
    except:
        return "Search unavailable right now."

# ============================================================
# INTENT DETECTION
# ============================================================

def rule_engine(user_input):
    for intent, data in intents.items():
        if user_input in data["patterns"]:
            return intent, 1.0
    for prefix in ("what is", "tell me about", "explain"):
        if user_input.startswith(prefix):
            topic = user_input.replace(prefix, "").strip()
            for intent, data in intents.items():
                if topic in data["patterns"]:
                    return intent, 1.0
            if topic:
                for intent, data in intents.items():
                    if topic in [p for p in data["patterns"] if len(p.split()) == 1]:
                        return intent, 0.95
            return None, 0.0
    return None, 0.0

@lru_cache(maxsize=128)
def semantic_search_cached(text):
    q = embedder.encode([text])[0]
    scores, idxs = intent_index.search(np.array([q], dtype=np.float32), 1)
    return intent_map[idxs[0][0]], float(scores[0][0])

def llm_fallback(raw, history):
    history_block = "\n".join(f"{m['role']}: {m['text']}" for m in history[-4:])
    prompt = (
        "You are DecodeBot, a friendly helpful assistant. "
        "Keep answers concise (2-3 sentences).\n\n"
        f"{history_block}\n\nUser: {raw}\nDecodeBot:"
    )
    try:
        return gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"(LLM error: {e})"

# ============================================================
# MAIN RESPONSE PIPELINE
# ⚠️  KEY FIX: tools get raw_input, intents get cleaned input
# ============================================================

def get_response(raw, history):
    raw_lower = raw.lower()

    # ── Tool layer (raw — operators preserved) ──
    if "weather" in raw_lower:
        return get_weather(raw), 1.0, "🌤 weather"

    calc = calculate(raw)                          # raw → 2+2 stays as 2+2
    if calc:
        return calc, 1.0, "🧮 calculator"

    if any(k in raw_lower for k in ["search", "look up", "find"]):
        if not any(x in raw_lower for x in ["ai", "python", "machine learning", "ml"]):
            return search_web(raw), 1.0, "🔍 search"

    # ── Intent layers (cleaned input) ──
    cleaned = clean(raw)

    intent, score = rule_engine(cleaned)
    if intent:
        return random.choice(intents[intent]["responses"]), score, "rule"

    intent, score = semantic_search_cached(cleaned)
    if score >= 0.70:
        return random.choice(intents[intent]["responses"]), score, "semantic"
    if score >= 0.45:
        return random.choice(intents[intent]["responses"]), score, "semantic-low"

    # ── LLM fallback (raw — full context) ──
    return llm_fallback(raw, history), score, "llm"

# ============================================================
# NAME HANDLER
# ============================================================

def handle_name(user_input):
    for phrase in ("my name is", "i am", "im", "call me"):
        if user_input.startswith(phrase):
            candidate = user_input.replace(phrase, "").strip()
            if candidate and len(candidate.split()) <= 2 and candidate.replace(" ", "").isalpha():
                return candidate.title()
    return None

# ============================================================
# SESSION STATE
# ============================================================

if "sessions"  not in st.session_state: st.session_state.sessions  = {"Chat 1": []}
if "active"    not in st.session_state: st.session_state.active    = "Chat 1"
if "user_name" not in st.session_state: st.session_state.user_name = None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-family:Playfair Display,serif;font-size:1.2rem;color:#3D2B2B;font-weight:600;margin-top:0.3rem;'>DecodeBot</div>
        <div style='font-size:0.72rem;color:#A08070;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;'>Your AI Assistant</div>
    </div>
    <hr style='border:none;height:1px;background:linear-gradient(to right,transparent,#C9967E,transparent);margin:0.5rem 0.5rem 1rem;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✦ New Chat"):
        new_name = f"Chat {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[new_name] = []
        st.session_state.active = new_name
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    search = st.text_input("Search", placeholder="🔍 Search chats...", key="search_box", label_visibility="collapsed")
    st.markdown("<div style='font-size:0.75rem;color:#A08070;letter-spacing:0.08em;text-transform:uppercase;margin:0.8rem 0 0.4rem 0.3rem;'>Chat History</div>", unsafe_allow_html=True)

    for sname in reversed(list(st.session_state.sessions.keys())):
        if search and search.lower() not in sname.lower():
            msgs = st.session_state.sessions[sname]
            if not any(search.lower() in m["text"].lower() for m in msgs):
                continue
        is_active = sname == st.session_state.active
        if st.button(f"{'💬 ' if is_active else '○ '}{sname}", key=f"btn_{sname}"):
            st.session_state.active = sname
            st.rerun()

    st.markdown("<div style='position:fixed;bottom:1.2rem;left:0;width:260px;text-align:center;font-size:0.72rem;color:#C9A898;'>Built by Nayab Nayyer 🌸</div>", unsafe_allow_html=True)

# ============================================================
# MAIN CHAT
# ============================================================

messages = st.session_state.sessions[st.session_state.active]

st.markdown("""
<div class="chat-header"><h1>🤖 DecodeBot</h1><p>Rule · Semantic · Gemini LLM</p></div>
<hr class="header-rule">
""", unsafe_allow_html=True)

def render_bubble(role, text, meta=None):
    if role == "user":
        st.markdown(f'<div class="bubble-row user"><div class="bubble user">{text}</div><div class="avatar user-av">🙂</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bubble-row bot"><div class="avatar bot-av">🤖</div><div class="bubble bot">{text}</div></div>', unsafe_allow_html=True)
        if meta:
            score, source = meta
            color = {
                "rule": "#2E7D52", "semantic": "#1565C0",
                "semantic-low": "#B07030", "llm": "#7A3A8A",
                "🌤 weather": "#0277BD", "🧮 calculator": "#6A1B9A",
                "🔍 search": "#E65100"
            }.get(source, "#888")
            st.markdown(f'<div class="debug-badge"><span style="color:{color}">◆ {source}</span>&nbsp;&nbsp;score: {score:.2f}</div>', unsafe_allow_html=True)

if not messages:
    name_display = st.session_state.user_name or "there"
    st.markdown(f"""
    <div class="welcome-box">
        <div style="font-size:3.5rem;">🌸</div>
        <p>Hi {name_display}! I'm DecodeBot.<br>
        Ask me about <strong>AI</strong>, <strong>Python</strong>, or <strong>Machine Learning</strong> —
        or try <strong>2 + 2</strong>, <strong>weather in Lahore</strong>, or anything else!</p>
    </div>
    <div class="chip-row">
        <span class="chip">What is AI?</span>
        <span class="chip">2 * 8</span>
        <span class="chip">Weather in Lahore</span>
        <span class="chip">Who are you?</span>
    </div>""", unsafe_allow_html=True)
else:
    for msg in messages:
        render_bubble(msg["role"], msg["text"], msg.get("meta"))

st.markdown('<div class="input-divider"></div>', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])
with col1:
    user_input_raw = st.text_input("msg", placeholder="Type a message...", key="input_box", label_visibility="collapsed")
with col2:
    send = st.button("Send ✦")

if send and user_input_raw.strip():
    raw = user_input_raw.strip()
    cleaned = clean(raw)

    name = handle_name(cleaned)
    if name:
        st.session_state.user_name = name
        messages.append({"role": "user",  "text": raw})
        messages.append({"role": "bot",   "text": f"Nice to meet you, {name}! 🌸 Ask me anything.", "meta": (1.0, "rule")})
    else:
        # ⚠️ KEY FIX: pass raw to get_response so tools get operators intact
        response, score, source = get_response(raw, messages)
        messages.append({"role": "user", "text": raw})
        messages.append({"role": "bot",  "text": response, "meta": (score, source)})

    # auto-rename chat from first message
    active = st.session_state.active
    if len(messages) == 2:
        short_name = raw[:28] + ("..." if len(raw) > 28 else "")
        st.session_state.sessions[short_name] = st.session_state.sessions.pop(active)
        st.session_state.active = short_name

    st.rerun()
