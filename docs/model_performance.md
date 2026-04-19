# 📊 Model Performance

## 🎯 Classification Models

| Model                  | Accuracy | Description                     |
|------------------------|----------|---------------------------------|
| 3-Month Placement      | 79.2%    | Short-term placement prediction |
| 6-Month Placement      | 83.1%    | Mid-term prediction             |
| 12-Month Placement     | 86.5%    | Long-term prediction            |

## 💰 Regression Model

| Model            | Metric | Value |
|------------------|--------|-------|
| Salary Predictor | RMSE   | ~1.5L |

---

## 🧠 Features Used

- CGPA  
- Number of Internships  
- College Tier  
- Job Domain  
- Location (Metro / Non-Metro)

---

## ⚙️ Model Configuration

- Algorithm: Random Forest
- Number of Trees: 50
- Max Depth: 10
- Random State: 42

---

## 🔍 Feature Importance

- CGPA → High impact  
- Internships → High impact  
- College Tier → Medium  
- Location → Low  