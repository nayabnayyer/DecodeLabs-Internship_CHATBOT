# 🤖 DecodeBot — Hybrid Conversational AI System

A hybrid conversational AI application built in Python that combines rule-based processing, semantic retrieval, and Large Language Model (LLM) generation into a multi-stage chatbot architecture.

DecodeBot explores how traditional NLP approaches and modern generative AI can work together to create reliable and flexible conversational systems.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-purple?style=flat-square&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-Hybrid_AI-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

# 🌟 Overview

Modern conversational AI systems require a balance between:

- predictable responses
- efficient retrieval
- flexible reasoning

DecodeBot implements a hybrid approach where user queries pass through multiple processing layers before generating a response.

Instead of relying only on an LLM, the system combines:

- deterministic rule-based responses
- semantic similarity search
- external tools
- Gemini LLM fallback generation

This creates a chatbot architecture that is both efficient and adaptable.

---

# ✨ Features

| Feature | Description |
|---|---|
| 🧠 Hybrid AI Pipeline | Combines rules, retrieval, and LLM generation |
| 🔍 Semantic Search | FAISS-based similarity matching for intent retrieval |
| ✨ Gemini Integration | Handles unknown and complex queries |
| 🔧 Tool Layer | Calculator, weather, and web search capabilities |
| 💬 Multi-turn Conversation | Maintains session-based chat history |
| 📝 Chat Management | Multiple conversations and chat search |
| 🎯 Confidence Routing | Selects response source based on similarity score |
| 🎨 Streamlit Interface | Interactive chatbot UI with custom styling |

---

# 🏗️ System Architecture

```
User Input
|
↓
🔧 Tool Layer
(Calculator / Weather / Search)
|
↓
📋 Rule Engine
(Deterministic Responses)
|
↓
🧠 Semantic Retrieval
(FAISS Similarity Search)
|
↓
✨ Gemini LLM
(Generative Response)
```

The system follows a layered decision process:

1. Handle structured requests with specialized tools
2. Use deterministic rules for known patterns
3. Retrieve relevant responses through semantic similarity
4. Use Gemini when the query requires broader reasoning

---

# 🔧 Key Technical Decisions

## Hybrid Routing Architecture

DecodeBot avoids sending every query directly to an LLM.

Instead:

- Simple queries are handled through fast deterministic methods
- Similar known queries are retrieved through semantic search
- Complex questions are forwarded to the LLM

This reduces unnecessary API calls while maintaining conversational flexibility.

---

## Semantic Retrieval

The retrieval component uses vector similarity search to identify relevant intents.

Workflow:

```
User Query
↓
Text Representation
↓
Vector Similarity Search
↓
Confidence Evaluation
↓
Response Selection
```

FAISS is used for efficient similarity matching between query vectors and stored intent patterns.

---

## Confidence-Based Routing

Response selection is based on similarity confidence:

```
High confidence
↓
Semantic response

Low confidence
↓
Fallback handling

Unknown query
↓
Gemini LLM generation
```

---

## LLM Fallback

When previous layers cannot confidently answer a query, Gemini generates a response using recent conversation context.

This provides flexibility for open-ended questions while maintaining controlled routing.

---

# 🚀 Features Demonstration

## Calculator

```
User:
2 * 8

Bot:
2 * 8 = 16
```

---

## Semantic Understanding

```
User:
Explain artificial intelligence

Bot:
AI is a field of computer science...
```

---

## Weather Tool

```
User:
Weather in Lahore

Bot:
Current weather information retrieved
```

---

## LLM Knowledge Query

```
User:
Who invented the internet?

Bot:
Generates contextual explanation through Gemini
```

---

# 🖥️ User Interface

Built using Streamlit with custom CSS.

Features include:

- Chat-style interface
- Sidebar navigation
- Multiple sessions
- Conversation history
- Response source indicators
- Interactive user experience

---

# 🧪 Testing

The system was tested across different query categories:

| Category | Example |
|-|-|
| Greetings | Hello, Hi |
| Knowledge Questions | Explain AI |
| Mathematical Queries | 2 + 2 |
| External Information | Weather requests |
| Unknown Questions | LLM fallback testing |

Testing focused on verifying correct routing between chatbot components.

---

# 📁 Project Structure


decodebot/
│
├── decodebot_final.py
│
├── README.md
│
└── requirements.txt


---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/decodebot.git

cd decodebot
```
Install Dependencies
```
pip install streamlit numpy faiss-cpu requests google-generativeai
```
Configure Gemini API
```
Set your API key:
```
Windows:
```
set GEMINI_API_KEY=your_key_here
```
Mac/Linux:
```
export GEMINI_API_KEY=your_key_here
Run Application
streamlit run decodebot_final.py
```
Open:
```
http://localhost:8501
```
# ⚠️ Limitations
Semantic retrieval currently uses lightweight vector representations.
Chat memory is session-based rather than permanently stored.
LLM responses depend on external API availability.
Large-scale benchmark evaluation has not yet been performed.
🔮 Future Improvements

# Possible improvements include:

Replace lightweight embeddings with transformer-based embeddings
Implement Retrieval-Augmented Generation (RAG)
Add persistent database storage
Evaluate chatbot performance using NLP metrics
Add voice interaction capabilities
Improve explainability of chatbot decisions
🧠 What I Learned

# Through this project, I developed experience in:

Designing hybrid AI architectures
Combining rule-based NLP with generative AI
Implementing semantic retrieval systems
Working with vector similarity search
Integrating Large Language Models into applications
Building interactive AI applications using Streamlit

## 👩‍💻 Author

**Nayab Nayyer**  
Fresh CS Graduate · Python · AI/ML · Streamlit  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/nayab-nayyer-2b6803321)

---

## 📄 License

Open source under the [MIT License](LICENSE).
