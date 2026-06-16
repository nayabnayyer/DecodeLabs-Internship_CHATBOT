# 🤖 DecodeBot — Hybrid AI Chatbot

A production-style conversational AI built from scratch in Python — combining rule-based NLP, semantic vector search, and a Gemini LLM fallback into one clean Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-purple?style=flat-square&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ What Is This?

DecodeBot is not a simple if-else chatbot. It's a **4-layer hybrid NLP system** that routes every message through an intelligent pipeline before deciding how to respond — the same architecture pattern used in production chatbots.

```
User Input
    ↓
🔧 Tool Layer        →  calculator / weather / web search
    ↓
📋 Rule Engine       →  exact match + prefix rules ("what is X")
    ↓
🧠 Semantic Search   →  FAISS vector similarity (offline embeddings)
    ↓
✨ Gemini LLM        →  real reasoning fallback for anything unknown
```

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🔧 **Calculator** | Evaluates `2 + 2`, `10 * 5`, `100 / 4` — numeric, not string concat |
| ☁️ **Live Weather** | Real-time weather via wttr.in — "weather in Lahore" |
| 🔍 **Web Search** | DuckDuckGo instant answers — "search for black holes" |
| 🧠 **Semantic Search** | FAISS + TF-IDF embeddings for meaning-aware intent matching |
| ✨ **Gemini Fallback** | Google Gemini 2.0 Flash answers anything the rules can't |
| 💬 **Multi-session** | Multiple chat sessions with sidebar navigation |
| 🔎 **Chat Search** | Search across all past chats by keyword |
| 📝 **Auto-rename** | Chats rename themselves from your first message |
| 🧠 **Name Memory** | Bot remembers your name for the whole session |
| 📊 **Debug Badges** | Every response shows its source layer + confidence score |

---

## 🏗️ Architecture Deep Dive

### Layer 1 — Tool Engine
Intercepts structured requests before any NLP runs. Receives **raw input** (symbols preserved) so `2 + 2` isn't mangled into `22` by text cleaning.

### Layer 2 — Rule Engine
Exact pattern matching + prefix rules. Fast, deterministic, zero latency. Handles `"what is python"`, `"tell me about AI"`, `"explain ML"` with `score=1.0`.

### Layer 3 — Semantic Search
Custom `SimpleEmbedder` builds a TF-IDF-style vocabulary from intent patterns, encodes them as normalized float vectors, and stores them in a **FAISS IndexFlatIP** (inner product = cosine similarity). Fully offline — no HuggingFace calls at runtime.

```python
# Confidence thresholds
score >= 0.70  →  respond with intent (high confidence)
score >= 0.45  →  respond with intent (low confidence note)
score < 0.45   →  escalate to Gemini LLM
```

### Layer 4 — Gemini LLM Fallback
When all layers fail, the last 4 turns of chat history are injected into a Gemini 2.0 Flash prompt for context-aware reasoning. Unknown inputs get real answers instead of "I don't understand."

---

## 🖥️ UI

Built with Streamlit + custom CSS — warm cream and rose gold theme.

- Chat bubbles (user right, bot left) with avatars
- Sidebar with session management, search, and New Chat
- Colour-coded debug badges per response:
  - 🟢 `rule` · 🔵 `semantic` · 🟠 `semantic-low` · 🟣 `llm` · 🔷 `weather` · 🟤 `calculator`

---

## ⚙️ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/decodebot.git
cd decodebot
```

### 2. Create a virtual environment
```bash
python -m venv chatbot_env
chatbot_env\Scripts\activate      # Windows
source chatbot_env/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install streamlit numpy faiss-cpu requests google-generativeai
```

### 4. Add your Gemini API key
Get a free key at **https://aistudio.google.com/**

Open `decodebot_final.py` and replace:
```python
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
```

Or set it as an environment variable (recommended):
```bash
set GEMINI_API_KEY=your_key_here        # Windows
export GEMINI_API_KEY=your_key_here     # Mac/Linux
```

### 5. Run
```bash
python -m streamlit run decodebot_final.py
```

Open **http://localhost:8501** 🎉

---

## 💬 Example Conversations

```
You:  hi
Bot:  Hello! 👋 How can I help you today?         [rule | 1.00]

You:  what is machine learning?
Bot:  ML is a subset of AI where systems learn...  [rule | 1.00]

You:  2 * 8
Bot:  🧮 2 * 8 = 16                               [calculator | 1.00]

You:  weather in Lahore
Bot:  ☁️ Weather in Lahore: Mist +29°C            [weather | 1.00] 

You:  who invented the internet?
Bot:  The internet evolved from ARPANET...         [llm | 0.21]

You:  my name is Nayab
Bot:  Nice to meet you, Nayab! 🌸 Ask me anything.
```

---

## 📁 Project Structure

```
decodebot/
│
├── decodebot_final.py    # Complete app — one file
└── README.md             # You're reading this
```

---

## 🛣️ Roadmap

- Persistent chat history (SQLite)
- User authentication
- Deploy to Streamlit Cloud
- Add more intents via JSON config (no code changes)
- Swap SimpleEmbedder for sentence-transformers when online
- Voice input support

---

## 🧠 What I Learned Building This

- Why text cleaning must be scoped — stripping symbols before tool detection breaks math
- How FAISS inner product search works as cosine similarity on normalized vectors
- The difference between rule-based, semantic, and LLM-based NLP — and when to use each
- How Streamlit session state enables multi-turn, multi-session conversation

---

## 👩‍💻 Author

**Nayab Nayyer**  
Fresh CS Graduate · Python · AI/ML · Streamlit  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/nayab-nayyer-2b6803321)

---

## 📄 License

Open source under the [MIT License](LICENSE).
