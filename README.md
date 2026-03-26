# 🤖 SentientBot: Real-Time Sentiment Analysis Chatbot

A full-stack AI chatbot that performs **Layered Sentiment Analysis** on user inputs to adapt its responses based on the speaker's emotional state.

## 🚀 Core Functionality
Unlike standard bots, SentientBot categorizes inputs into **Positive, Neutral, and Negative** tiers, while also identifying secondary emotions (Frustration, Urgency, Joy) to adjust the "Tone" of the reply.

## 🛠 Tech Stack
- **Frontend:** React + Tailwind CSS (Real-time message streaming).
- **Backend:** Python (FastAPI) for high-performance asynchronous processing.
- **ML Engine:** HuggingFace Transformers (`distilbert-base-uncased-finetuned-sst-2-english`).
- **Database:** SQLAlchemy + SQLite (Storing conversation history for trend analysis).

## 🧠 ML Logic & Pipeline
1. **Preprocessing:** Tokenization and cleaning of raw string data.
2. **Inference:** The model calculates a probability distribution across sentiment labels.
3. **Thresholding:** If confidence is below 0.7, the bot requests clarification to ensure accuracy.

## 📂 Project Structure
```bash
├── backend/
│   ├── main.py          # FastAPI routes
│   ├── model.py         # Sentiment analysis inference logic
│   └── database.py      # SQLAlchemy models
├── frontend/
│   ├── src/             # React components
│   └── App.js           # Chat interface
└── README.md
