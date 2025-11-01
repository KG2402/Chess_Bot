# ♟️ Chess Q&A Chatbot

AI-powered chess assistant built with Streamlit and LangChain, featuring intelligent guardrails, conversation memory, and personalization.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-00A67E?style=for-the-badge&logo=groq&logoColor=white)

## 🚀 Features

### 🤖 AI-Powered Chess Expertise
- **Llama 3.1 8B Instant** via Groq API for lightning-fast responses
- **Comprehensive chess knowledge**: rules, strategies, openings, endgames, famous players, and history
- **Standard algebraic notation** for move explanations

### 🛡️ Intelligent Guardrails
- **Chess-only content filtering** to ensure topic relevance
- **Keyword and pattern matching** for chess-related queries
- **Graceful rejection** of non-chess questions with helpful guidance

### 🧠 Conversation Memory
- **Context-aware responses** using LangChain memory
- **10-message conversation buffer** for coherent dialogue
- **Persistent session state** throughout the chat

### 👤 Personalization
- **Automatic name extraction** from introductions
- **Personalized greetings** using user's name
- **Customized responses** based on conversation context

### 💾 Export & Management
- **Download full conversation history** as text files
- **Session statistics** tracking questions and messages
- **One-click chat reset** for fresh conversations

## 📋 Prerequisites

- Python 3.8+
- Groq API account ([Get free API key](https://console.groq.com/keys))
- Basic understanding of chess terminology
