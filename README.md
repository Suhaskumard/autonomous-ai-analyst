# 🚀 Autonomous AI Analyst

<p align="center">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" />
</p>

An intelligent, full-stack data analytics platform that acts as an autonomous AI companion. It transcends traditional dashboards by providing live analysis, AI-driven conversational insights, and smart data recommendations.

## ✨ High-Impact Features

### 🧠 1. AI Copilot (Chat with Data)
A powerful LLM-powered chat interface anchored directly to your uploaded datasets.
- **Context-Aware:** Ask questions like *"Why is my model accuracy low?"* or *"Which features are most important?"*
- **Smart Suggestions:** One-click suggested prompts for immediate deep dives.

### 📊 2. Live Analysis Pipeline
Experience real-time transparency during data crunching.
- **Dynamic Progress:** Watch the engine step through Data Cleaning, Model Training, and Evaluation.
- **Live Terminal Specs:** A simulated streaming log feed outputs internal model variations (e.g., `Training RandomForest... Accuracy: 78% → 85%`).

### 💡 3. Smart Recommendation Engine
Proactive, data-driven suggestions rendered straight to the dashboard. 
- Identifies missing value anomalies.
- Recommends algorithm swaps (e.g., suggesting XGBoost for potential accuracy bumps).

## 🏗️ Architecture

The system maintains a clean separation of concerns:

- **Frontend (React / Vite):** Highly optimized, scalable component architecture leveraging custom CSS variables for premium Glassmorphism and dark-mode aesthetics.
- **Backend (FastAPI):** Asynchronous Python backend designed to handle ML payloads, serve predictions, and orchestrate automated insights generation.

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (for backend)
- **Node.js 20+** and **npm** (for frontend)
- **Git** (optional, for future updates)

### 1. Backend Setup
Open a terminal and navigate to the backend directory:

```bash
cd backend
```

Create and activate a virtual environment (Windows/Linux/macOS compatible):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the API server:
```bash
uvicorn main:app --reload --port 8000
```

Backend will be available at http://localhost:8000/docs (Swagger UI).

### 2. Frontend Setup (New Terminal)
Open a new terminal and navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Frontend dev server starts at http://localhost:5173 (automatically proxies backend API).

### 3. Access the App
- Open http://localhost:5173 in your browser.
- Upload CSV data and explore AI insights!

## 📁 Project Structure
```
autonomous-ai-analyst/
├── backend/          # FastAPI server
│   ├── main.py
│   ├── requirements.txt
│   └── routes/       # API endpoints
├── frontend/         # React + Vite
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── README.md         # This file
└── TODO.md
```

## 🔧 Production Build
```bash
# Backend: Deploy with gunicorn/uvicorn (e.g., on Railway/Render)
# Frontend:
cd frontend
npm run build    # Creates /dist folder
# Serve dist/ with nginx/Vercel/Netlify
```  

## 🐛 Troubleshooting
- **Port conflicts**: Change `--port 8000` or kill processes on ports.
- **CORS errors**: Backend allows all origins by default.
- **Python deps fail**: Ensure venv activated, `pip list` shows FastAPI.
- **Frontend proxy fails**: Check Network tab; backend must run first.
- **Windows venv**: Use `where python` to confirm venv Python.

## 🤝 Contributing
1. Fork/clone repo.
2. Create feature branch.
3. Follow setup above.
4. Submit PR.

---
*Built as a next-generation SaaS project.*
