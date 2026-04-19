# 🚀 Deployment Guide

## 🌐 Option 1: Streamlit Cloud (Recommended)

### Steps:
1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Select `streamlit_app.py`
5. Click Deploy

### Output:
- Live web app URL
- Accessible globally

---

## 💻 Option 2: Local Deployment

### Install dependencies:
```bash
pip install pandas numpy scikit-learn streamlit plotly groq
```
### Run application:
```bash
streamlit run streamlit_app.py
```

### Access:
http://localhost:8501

## ☁️ Option 3: Docker (Advanced)

### Build:
```bash
docker build -t placement-ai .
```

### Run:
```bash
docker run -p 8501:8501 placement-ai
```