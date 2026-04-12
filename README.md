# 🚀 ADVANCED AUTONOMOUS AI ANALYST

An enterprise-grade, full-stack autonomous data science system. It automatically ingests tabular data, executes an end-to-end ML pipeline, and provides an **Advanced AI Analyst** powered by Gemini with native **Code Execution** capabilities.

![Aesthetic Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphism-blueviolet)
![Engine](https://img.shields.io/badge/Engine-Gemini_1.5_Flash-orange)
![Capability](https://img.shields.io/badge/Chat-Code_Execution_Enabled-success)

## 🌟 Key Features

- **Advanced AI Chatbot (Gemini-Powered)**:
  - **Native Code Execution**: The AI writes and runs real Python code on your datasets to calculate correlations, find outliers, and generate custom insights.
  - **Contextual Memory**: Remembers past interactions for multi-turn analytical deep-dives.
  - **Dynamic Analysis**: Ask "Find the top 3 correlations" or "Plot the distribution of the target," and the AI handles the logic.
- **Premium Design System**: 
  - Modern **Glassmorphic** UI with a sophisticated dark-mode aesthetic.
  - HSL-tailored color palette with vibrant accents and micro-interactions.
  - Responsive layout with high-performance animations.
- **Autonomous ML Pipeline**: 
  - **Zero-touch Preprocessing**: Automatic scaling, encoding, and target detection.
  - **Smart Modeling**: Auto-selection, Ensembling, or Manual fine-tuning.
  - **Explainability**: Integrated SHAP-based feature importance.
- **Performance Caching**:
  - SHA256 dataset hashing ensures instant reuse of previously trained models & insights.

---

## 🛠 Project Architecture

```text
backend/
  routes/
    chat.py       <-- Advanced Gemini Code Execution Engine
    upload.py     <-- Pipeline orchestration & Caching
    predict.py    <-- Inference API
  ml/             <-- Core Analytical Engine
    train.py, preprocess.py, explain.py, quality.py
  models/
    metadata/     <-- Persistent dataset blueprints
    saved_models/ <-- ML Artifacts

frontend/
  src/
    styles.css    <-- Premium Design System
  components/     <-- Modern React Components (Lucide-powered)
  pages/          <-- Unified Analytical Workflows
```

---

## 🚀 Quick Start

### 1. Setup Backend
1. Navigate to `backend/`.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`.
4. **Configure Environment Variables**:
   ```bash
   export GEMINI_API_KEY="your_api_key"
   export GEMINI_MODEL="gemini-1.5-flash"
   ```
5. Run the server: `uvicorn main:app --reload`.

### 2. Setup Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Start the dev server: `npm run dev`.

---

## 🤖 Using the Advanced Analyst

Once a dataset is uploaded, the **Advanced AI Analyst** is your primary partner for exploration:

- **Mathematical Proof**: "What is the standard deviation of the target relative to feature X?"
- **Complex Logic**: "Filter the dataset for rows where X > 50 and then calculate the mean of Y."
- **Visual Description**: "Run a correlation analysis and tell me which features have a high coefficient."

The Analyst doesn't just "guess"—it writes **Python code** using `pandas` to provide ground-truth answers from your specific CSV snapshot.

---

## 🎨 Design Philosophy

The interface follows a **"Depth & Clarity"** approach:
- **Glassmorphism**: Layers of transparency and blur for a hierarchy of information.
- **Electric Accents**: Indigo and Pink accents provide high-contrast visual cues.
- **Micro-interactions**: Hover states, smooth progress transitions, and slide-up animations for a premium feel.

---

## 📋 Operational Notes

- **Clean Slate**: This upgrade identifies and removes legacy `.pkl` files to ensure a fresh, consistent environment.
- **Dataset Snapshot**: The backend stores a sanitized CSV snapshot for the Chatbot to execute code against.
- **Privacy**: Code execution runs in a sandboxed environment provided by Google Gemini.
