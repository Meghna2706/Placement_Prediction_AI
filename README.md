# 🎓 Placement Prediction AI  
### ML-Powered Employability Forecasting & Career Intelligence System

---

## 🎯 Overview

**Placement Prediction AI** is an end-to-end machine learning system that predicts a student’s likelihood of securing a job across multiple timelines (3, 6, and 12 months), along with expected salary and risk level.

The system is designed for:
- 🏦 Financial institutions (education loans & risk assessment)  
- 🎓 EdTech platforms (student guidance & analytics)  
- 🏫 Universities (placement performance tracking)

It combines **machine learning + AI-driven career coaching** to deliver both predictions and actionable insights.

---

## 🚀 Live Demo

🔗 https://placementpredictionai.streamlit.app/

### 📸 Key Screens
- Student Profile Input Dashboard  
- Placement Probability Metrics  
- Risk Assessment Gauge  
- AI Career Recommendations  
- Placement Timeline Visualization  

---

## 📊 Performance Metrics

- 🎯 Placement Prediction Accuracy:
  - 3 Months: **79.2%**
  - 6 Months: **83.1%**
  - 12 Months: **86.5%**
- ⚡ Inference Time: **<100ms**
- 🧠 Model Type: **Random Forest Ensemble**
- 📦 Dataset: **Synthetic (500+ samples)**

> ⚠️ Note: Metrics are based on controlled synthetic data simulation.

---

## ✨ Key Features

- 📊 Multi-horizon placement prediction (3, 6, 12 months)  
- 💰 Salary estimation using regression modeling  
- ⚠️ Risk classification (Low / Medium / High)  
- 📈 Interactive dashboards using Plotly  
- 🤖 AI-powered career recommendations (Groq LLaMA 3)  
- ⚡ Real-time predictions with minimal latency  

---

## 🧬 System Architecture

### 🔹 Input Features
- CGPA  
- Number of Internships  
- College Tier  
- Job Domain  
- Location (Metro / Non-Metro)

### 🔹 Models Used
- Random Forest Classifier (3 models for time-based prediction)  
- Random Forest Regressor (salary estimation)

### 🔹 Pipeline Flow
User Input → Feature Encoding → Model Inference → Risk Analysis → AI Suggestions

---

## 🎬 Example Predictions

### ✅ Strong Profile
- CGPA: 8.5 | Tier 1 | IT | 3 Internships | Metro  
- 📈 3 Months: 80%  
- 📈 6 Months: 95%  
- 💰 Salary: ₹16L  
- ⚠️ Risk: LOW  
- ✔ Recommendation: APPROVE  

---

### ⚖️ Average Profile
- CGPA: 7.0 | Tier 2 | Internship | Metro  
- 📈 3 Months: 60%  
- 📈 6 Months: 78%  
- 💰 Salary: ₹12L  
- ⚠️ Risk: MEDIUM  
- ✔ Recommendation: APPROVE WITH CONDITIONS  

---

### 🔴 At-Risk Profile
- CGPA: 6.0 | Tier 3 | No Internship | Non-Metro  
- 📈 3 Months: 40%  
- 📈 6 Months: 55%  
- 💰 Salary: ₹9L  
- ⚠️ Risk: HIGH  
- ❗ Recommendation: SKILL-UP FIRST  

---

## 🧰 Tech Stack

- Python  
- scikit-learn  
- Streamlit  
- Plotly  
- Groq API (LLaMA 3)  
- NumPy & Pandas  

---

## 🛠️ Local Setup (For Developers)

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn streamlit plotly groq
```

### 2. Run the Application
```bash
streamlit run streamlit_app.py
```

App will open at:


http://localhost:8501


---

## 🚀 Deployment Options

### 🌐 Streamlit Cloud (Recommended)
- Free deployment  
- GitHub integration  
- Auto updates  

### ☁️ Cloud Platforms
- AWS (EC2 / Cloud Run)  
- Google Cloud Platform  
- Docker-based deployment  

---

## ⚠️ Limitations

- Model is trained on synthetic data (not real-world datasets)  
- Salary predictions are heuristic-based approximations  
- AI suggestions depend on prompt quality and LLM behavior  

---

## 🚀 Future Improvements

- 📄 Resume parsing (PDF → structured features)  
- 📊 Real-world dataset integration  
- 🏢 Company-specific hiring predictions  
- 🎯 Personalized skill roadmap generation  
- 📡 Live job market integration  

---

## 📁 Project Structure
```
Placement_Prediction_AI/
│
├── streamlit_app.py              # Streamlit web app
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
│
├── output/                       # Generated outputs (predictions, logs, etc.)
│
├── data/                         # Dataset & related files
│   └── data_dictionary.md
│
├── docs/                         # Supporting documentation
│   ├── architecture_diagram.png
│   ├── model_performance.md
│   ├── deployment_guide.md
│   └── problem_statement.md
```
---

## 👩‍💻 Author

**Subramani Meghna**

---

## 💡 Final Note

This project demonstrates how **machine learning + AI** can transform traditional placement 
