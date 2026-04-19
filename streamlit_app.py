"""
PLACEMENT PREDICTION SYSTEM - GROQ AI POWERED STREAMLIT APP (COMPLETE & FIXED)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import sys
import importlib

# ============================================================================
# FORCE RELOAD
# ============================================================================
print("🔄 Reloading modules...")
if 'models_cache' in st.session_state:
    del st.session_state['models_cache']

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Placement Predictor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GROQ API KEY SETUP
# ============================================================================
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""

with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    api_key = st.text_input("Enter your Groq API key:", 
                          type="password", 
                          value=st.session_state.groq_api_key,
                          help="Get FREE key: https://console.groq.com/keys")
    
    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("✅ Groq API Key set!")
    else:
        st.warning("⚠️ Enter key for AI suggestions")
    
    st.markdown("---")

def get_groq_client():
    if st.session_state.groq_api_key:
        return Groq(api_key=st.session_state.groq_api_key)
    return None

def get_ai_suggestions(student_data, predictions):
    client = get_groq_client()
    
    domain = student_data['job_domain']
    
    # ---------------- FALLBACK (DOMAIN-SPECIFIC) ----------------
    fallback_map = {
        "IT": [
            "Master DSA and solve 200+ problems on LeetCode",
            "Build 2-3 full-stack projects and deploy them",
            "Prepare system design basics for interviews",
            "Contribute to GitHub and optimize your resume"
        ],
        "Healthcare": [
            "Learn healthcare data tools (EHR, clinical data systems)",
            "Gain certifications in health informatics or bioinformatics",
            "Work on real-world healthcare datasets/projects",
            "Understand regulations like HIPAA and patient data ethics"
        ],
        "Manufacturing": [
            "Learn core concepts like supply chain & operations",
            "Gain hands-on experience with tools like AutoCAD / SolidWorks",
            "Understand lean manufacturing & Six Sigma basics",
            "Do internships in production or plant environments"
        ],
        "BFSI": [
            "Learn financial modeling and Excel advanced skills",
            "Understand banking products and risk management",
            "Get certifications like NISM or CFA Level 1 basics",
            "Stay updated with financial markets and trends"
        ],
        "Consulting": [
            "Practice case studies (market sizing, profitability)",
            "Improve structured problem-solving skills",
            "Build strong communication & presentation skills",
            "Learn basic business frameworks (SWOT, Porter's Five Forces)"
        ]
    }

    if not client:
        return fallback_map.get(domain, fallback_map["IT"])

    try:
        prompt = f"""
You are a strict career coach.

Student Profile:
- CGPA: {student_data['cgpa']:.1f}
- Internships: {student_data['num_internships']}
- College Tier: {student_data['college_tier']}
- Location: {"Metro" if student_data['is_metro_based'] else "Non-Metro"}
- Target Domain: {domain}

Predictions:
- 3 months: {predictions['prob_3mo']:.1f}%
- 6 months: {predictions['prob_6mo']:.1f}%
- Salary: ₹{predictions['salary']/100000:.1f}L
- Risk: {predictions['risk_level']}

INSTRUCTIONS:
- Give EXACTLY 4 actionable steps
- Each step MUST be specific to {domain}
- DO NOT give generic advice like "learn coding", "network", "practice interviews"
- Be practical and industry-relevant
- Keep each step under 12 words
- Numbered list only (1 to 4)

OUTPUT FORMAT:
1. ...
2. ...
3. ...
4. ...
"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=200,
            temperature=0.5
        )

        response_text = chat_completion.choices[0].message.content.strip()

        suggestions = []
        for line in response_text.split('\n'):
            cleaned = line.strip()
            if cleaned and any(cleaned.startswith(f"{i}.") for i in range(1, 5)):
                cleaned = cleaned.split('.', 1)[1].strip()
                suggestions.append(cleaned)

        # Safety fallback if model messes up
        if len(suggestions) < 4:
            return fallback_map.get(domain, fallback_map["IT"])

        return suggestions[:4]

    except Exception:
        return fallback_map.get(domain, fallback_map["IT"])
    
# ============================================================================
# TRAIN MODELS
# ============================================================================
@st.cache_resource(show_spinner=False)
def train_models():
    print("📊 Training models...")
    np.random.seed(42)
    
    n = 500
    data = {
        'cgpa': np.random.normal(7.2, 1.0, n).clip(5.0, 10.0),
        'num_internships': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
        'college_tier': np.random.choice([1, 2, 3], n, p=[0.2, 0.3, 0.5]),
        'job_domain': np.random.choice(['IT', 'BFSI', 'Manufacturing', 'Healthcare', 'Consulting'], n),
        'is_metro_based': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    }
    df = pd.DataFrame(data)
    
    domain_multipliers = {'IT': 1.1, 'BFSI': 1.0, 'Consulting': 1.05, 'Manufacturing': 0.85, 'Healthcare': 0.9}
    domain_mult_list = [domain_multipliers[domain] for domain in df['job_domain']]
    
    base_prob = 0.3 * (df['cgpa'] / 10) + 0.25 * (df['num_internships'] / 5) + 0.2 * (1 - df['college_tier']/3) + 0.15 * df['is_metro_based']
    placement_3mo_prob = np.clip(base_prob * domain_mult_list, 0.1, 0.95)
    
    df['placed_3mo'] = (np.random.random(n) < placement_3mo_prob).astype(int)
    df['placed_6mo'] = df['placed_3mo'] | (np.random.random(n) < placement_3mo_prob + 0.15).astype(int)
    df['placed_12mo'] = df['placed_6mo'] | (np.random.random(n) < placement_3mo_prob + 0.25).astype(int)
    
    domain_salaries = {'IT': 15, 'BFSI': 14, 'Consulting': 16, 'Manufacturing': 11, 'Healthcare': 12}
    domain_salary_list = [domain_salaries[domain] for domain in df['job_domain']]
    
    df['salary'] = np.clip(
        domain_salary_list + (df['cgpa'] - 5) * 0.8 + (df['num_internships'] / 5) * 2.5 + (3 - df['college_tier']) * 1.2 + np.random.normal(0, 1, n), 
        8, 22
    ) * 100000
    
    feature_cols = ['cgpa', 'num_internships', 'college_tier', 'is_metro_based']
    le = LabelEncoder()
    df['job_domain_encoded'] = le.fit_transform(df['job_domain'])
    feature_cols.append('job_domain_encoded')
    
    X = df[feature_cols]
    
    model_3mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_3mo.fit(X, df['placed_3mo'])
    model_6mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_6mo.fit(X, df['placed_6mo'])
    model_12mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_12mo.fit(X, df['placed_12mo'])
    model_salary = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model_salary.fit(X, df['salary'])
    
    print("✅ Models trained!")
    return {
        'model_3mo': model_3mo, 'model_6mo': model_6mo, 'model_12mo': model_12mo,
        'model_salary': model_salary, 'feature_cols': feature_cols, 'le': le
    }

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_prediction(student_data, models):
    le, feature_cols = models['le'], models['feature_cols']
    X = np.array([
        student_data['cgpa'], student_data['num_internships'], student_data['college_tier'],
        student_data['is_metro_based'], le.transform([student_data['job_domain']])[0]
    ]).reshape(1, -1)
    
    prob_3mo = models['model_3mo'].predict_proba(X)[0][1] * 100
    prob_6mo = models['model_6mo'].predict_proba(X)[0][1] * 100
    prob_12mo = models['model_12mo'].predict_proba(X)[0][1] * 100
    salary_pred = models['model_salary'].predict(X)[0]
    
    risk_level = "LOW" if prob_3mo > 70 and prob_6mo > 80 else "MEDIUM" if prob_6mo > 60 else "HIGH"
    risk_score = 15 if risk_level == "LOW" else 45 if risk_level == "MEDIUM" else 75
    
    strengths = []
    risks = []
    
    if student_data['cgpa'] >= 8: strengths.append("Excellent CGPA (8+)")
    if student_data['num_internships'] >= 2: strengths.append(f"{student_data['num_internships']}+ internships")
    elif student_data['num_internships'] == 1: strengths.append("1 internship - good start")
    if student_data['college_tier'] == 1: strengths.append("Top-tier college")
    if student_data['is_metro_based']: strengths.append("Metro location")
    
    if student_data['cgpa'] < 6.5: risks.append("Low CGPA - improve academics")
    if student_data['num_internships'] == 0: risks.append("No internships - critical!")
    elif student_data['num_internships'] == 1: risks.append("Need more internships")
    if student_data['college_tier'] == 3: risks.append("Tier-3 college - network more")
    if not student_data['is_metro_based']: risks.append("Non-metro - consider relocation")
    
    return {
        'prob_3mo': prob_3mo, 'prob_6mo': prob_6mo, 'prob_12mo': prob_12mo,
        'salary': salary_pred, 'risk_level': risk_level, 'risk_score': risk_score,
        'strengths': strengths, 'risks': risks
    }

# ============================================================================
# MAIN APP
# ============================================================================
st.title("🎓 Placement Prediction AI")
st.subheader("ML + Groq AI Career Coach")

if not st.session_state.groq_api_key:
    st.warning("🔑 Get FREE Groq API: https://console.groq.com/keys")

models = train_models()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📊 Student Profile")
    st.markdown("---")
    
    cgpa = st.slider("📈 CGPA", 5.0, 10.0, 7.5, 0.1)
    num_internships = st.slider("💼 Number of Internships", 0, 5, 1, 1)
    
    college_tier_text = st.selectbox("🏫 College Tier", 
        ["Tier 1 (IIT, BITS, etc.)", "Tier 2 (Mid-tier)", "Tier 3 (Other colleges)"], 1)
    college_tier = 1 + ["Tier 1 (IIT, BITS, etc.)", "Tier 2 (Mid-tier)", "Tier 3 (Other colleges)"].index(college_tier_text)
    
    job_domain = st.selectbox("🎯 Job Domain", 
        ["IT", "BFSI", "Manufacturing", "Healthcare", "Consulting"])
    
    is_metro_text = st.selectbox("📍 Location", 
        ["Non-Metro", "Metro (Delhi, Mumbai, Bangalore)"], 1)
    is_metro = 1 if "Metro" in is_metro_text else 0

student_data = {
    'cgpa': cgpa, 'num_internships': num_internships,
    'college_tier': college_tier, 'job_domain': job_domain,
    'is_metro_based': is_metro
}

result = make_prediction(student_data, models)
ai_suggestions = get_ai_suggestions(student_data, result)

with col2:
    st.markdown("### 🎯 Predictions")
    st.markdown("---")
    
    m1, m2 = st.columns(2)
    with m1: st.metric("3 months", f"{result['prob_3mo']:.1f}%")
    with m2: st.metric("6 months", f"{result['prob_6mo']:.1f}%")
    
    m3, m4 = st.columns(2)
    with m3: st.metric("12 months", f"{result['prob_12mo']:.1f}%")
    with m4: st.metric("Salary", f"₹{result['salary']/100000:.1f}L")

st.markdown("---")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📈 Risk Assessment")
    risk_colors = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}
    st.markdown(f"## {risk_colors[result['risk_level']]} {result['risk_level']} Risk")
    st.markdown(f"**Risk Score:** {result['risk_score']}%")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=result['risk_score'],
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': 'darkblue'},
               'steps': [{'range': [0, 33], 'color': "#90EE90"},
                        {'range': [33, 66], 'color': "#FFD700"},
                        {'range': [66, 100], 'color': "#FF6B6B"}]}
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_right:
    st.markdown("### 💡 Key Insights")
    if result['strengths']:
        st.markdown("**✅ Strengths:**")
        for s in result['strengths']: st.markdown(f"- {s}")
    if result['risks']:
        st.markdown("**⚠️ Improve:**")
        for r in result['risks']: st.markdown(f"- {r}")

st.markdown("---")
st.markdown("### 🤖 Groq AI Career Coach")

if st.session_state.groq_api_key:
    st.info("✨ Lightning-fast AI suggestions!")
else:
    st.info("🔑 Enter Groq API key for personalized advice")

st.markdown("**💪 Next Steps:**")
for i, suggestion in enumerate(ai_suggestions, 1):
    st.markdown(f"{i}. {suggestion}")

st.markdown("---")
st.markdown("### 📊 Placement Timeline")

fig_timeline = px.bar(
    {'Timeline': ['3 Months', '6 Months', '12 Months'], 
     'Probability': [result['prob_3mo'], result['prob_6mo'], result['prob_12mo']]},
    x='Timeline', y='Probability', color='Probability',
    color_continuous_scale=['#FF6B6B', '#FFD700', '#90EE90'],
    height=350
)
fig_timeline.update_layout(title="Job Timeline", showlegend=False)
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 ML + Groq Llama3 | Domain-specific predictions & AI coaching</p>
</div>
""", unsafe_allow_html=True)