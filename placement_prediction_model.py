"""
Placement Risk Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# STEP 1: CREATE SYNTHETIC DATASET (500 students)
# ============================================================================

def create_synthetic_data(n_samples=500):
    """Generate realistic student placement data"""
    np.random.seed(42)
    
    data = {
        'student_id': [f'STU{i:04d}' for i in range(n_samples)],
        'cgpa': np.random.normal(7.2, 1.0, n_samples).clip(5.0, 10.0),
        'has_internship': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'internship_duration_months': np.random.choice([0, 2, 3, 4, 6], n_samples),
        'college_tier': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'job_domain': np.random.choice(['IT', 'BFSI', 'Manufacturing', 'Healthcare', 'Consulting'], n_samples),
        'is_metro_based': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic outcome: Good CGPA + internship = higher chance of placement
    placement_3mo_prob = (
        0.3 * (df['cgpa'] / 10) +
        0.2 * df['has_internship'] +
        0.15 * (1 - df['college_tier']/3) +
        0.15 * df['is_metro_based'] +
        0.2 * (df['internship_duration_months'] / 6)
    )
    placement_3mo_prob = np.clip(placement_3mo_prob, 0.1, 0.95)
    df['placed_3mo'] = (np.random.random(n_samples) < placement_3mo_prob).astype(int)
    df['placed_6mo'] = df['placed_3mo'] | (np.random.random(n_samples) < placement_3mo_prob + 0.15).astype(int)
    df['placed_12mo'] = df['placed_6mo'] | (np.random.random(n_samples) < placement_3mo_prob + 0.25).astype(int)
    
    # Salary estimation: CGPA, internship, college tier influence
    base_salary = 12
    df['salary'] = (
        base_salary +
        (df['cgpa'] - 5) * 0.8 +
        df['has_internship'] * 1.5 +
        (3 - df['college_tier']) * 1.2 +
        np.random.normal(0, 1, n_samples)
    )
    df['salary'] = np.clip(df['salary'], 8, 22) * 100000  # 8L to 22L
    
    return df

# ============================================================================
# STEP 2: BUILD ML MODELS
# ============================================================================

def train_placement_models(df):
    """Train 3 classification models + 1 regression model"""
    
    # Prepare features
    feature_cols = ['cgpa', 'has_internship', 'internship_duration_months', 
                    'college_tier', 'is_metro_based']
    
    # Encode categorical features
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['job_domain_encoded'] = le.fit_transform(df['job_domain'])
    feature_cols.append('job_domain_encoded')
    
    X = df_encoded[feature_cols]
    
    # Train 3mo placement classifier
    y_3mo = df['placed_3mo']
    model_3mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_3mo.fit(X, y_3mo)
    
    # Train 6mo placement classifier
    y_6mo = df['placed_6mo']
    model_6mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_6mo.fit(X, y_6mo)
    
    # Train 12mo placement classifier
    y_12mo = df['placed_12mo']
    model_12mo = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model_12mo.fit(X, y_12mo)
    
    # Train salary regression model
    y_salary = df['salary']
    model_salary = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model_salary.fit(X, y_salary)
    
    # Calculate and print accuracies
    print("\n" + "="*60)
    print("MODEL ACCURACIES")
    print("="*60)
    print(f"Placement in 3 months: {accuracy_score(y_3mo, model_3mo.predict(X)):.2%}")
    print(f"Placement in 6 months: {accuracy_score(y_6mo, model_6mo.predict(X)):.2%}")
    print(f"Placement in 12 months: {accuracy_score(y_12mo, model_12mo.predict(X)):.2%}")
    
    return {
        'model_3mo': model_3mo,
        'model_6mo': model_6mo,
        'model_12mo': model_12mo,
        'model_salary': model_salary,
        'feature_cols': feature_cols,
        'le': le,
        'feature_importance': model_3mo.feature_importances_
    }

# ============================================================================
# STEP 3: MAKE PREDICTIONS WITH EXPLANATIONS
# ============================================================================

def predict_placement(student_data, models):
    """
    Make prediction for a single student with explanation
    
    student_data = {
        'cgpa': 8.5,
        'has_internship': 1,
        'internship_duration_months': 3,
        'college_tier': 1,
        'job_domain': 'IT',
        'is_metro_based': 1
    }
    """
    
    # Prepare features
    le = models['le']
    feature_cols = models['feature_cols']
    X = np.array([
        student_data['cgpa'],
        student_data['has_internship'],
        student_data['internship_duration_months'],
        student_data['college_tier'],
        student_data['is_metro_based'],
        le.transform([student_data['job_domain']])[0]
    ]).reshape(1, -1)
    
    # Get predictions
    prob_3mo = models['model_3mo'].predict_proba(X)[0][1]
    prob_6mo = models['model_6mo'].predict_proba(X)[0][1]
    prob_12mo = models['model_12mo'].predict_proba(X)[0][1]
    salary_pred = models['model_salary'].predict(X)[0]
    
    # Risk assessment: Low if all probabilities > 70%, High if any < 50%
    if prob_3mo > 0.7 and prob_6mo > 0.8:
        risk_level = "LOW"
        risk_score = 0.15
    elif prob_6mo > 0.6:
        risk_level = "MEDIUM"
        risk_score = 0.45
    else:
        risk_level = "HIGH"
        risk_score = 0.75
    
    # Explanation
    strengths = []
    risks = []
    
    if student_data['cgpa'] >= 8:
        strengths.append("Excellent CGPA")
    if student_data['has_internship'] == 1:
        strengths.append("Relevant internship experience")
    if student_data['college_tier'] == 1:
        strengths.append("Top-tier college")
    if student_data['is_metro_based'] == 1:
        strengths.append("Located in metro area with high job density")
    
    if student_data['cgpa'] < 6.5:
        risks.append("Lower than average CGPA")
    if student_data['has_internship'] == 0:
        risks.append("No internship experience")
    if student_data['college_tier'] == 3:
        risks.append("Tier-3 college - smaller recruiter pool")
    if student_data['is_metro_based'] == 0:
        risks.append("Non-metro location - fewer job opportunities")
    
    return {
        'placement_3mo_probability': round(prob_3mo * 100, 1),
        'placement_6mo_probability': round(prob_6mo * 100, 1),
        'placement_12mo_probability': round(prob_12mo * 100, 1),
        'expected_salary': int(salary_pred),
        'salary_range_min': int(salary_pred * 0.85),
        'salary_range_max': int(salary_pred * 1.15),
        'risk_level': risk_level,
        'default_probability': round(risk_score * 100, 1),
        'strengths': strengths,
        'risks': risks,
        'recommendations': [
            "Focus on coding interviews" if student_data['cgpa'] < 8 else "Excellent prepared",
            "Network with alumni in target companies",
            "Practice mock interviews regularly",
            "Stay updated on industry trends"
        ]
    }

# ============================================================================
# STEP 4: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PLACEMENT PREDICTION SYSTEM - HACKATHON BUILD")
    print("="*60)
    
    # Create synthetic data
    print("\n[1/4] Creating synthetic student dataset...")
    df = create_synthetic_data(n_samples=500)
    print(f"✓ Created {len(df)} student records")
    
    # Train models
    print("\n[2/4] Training ML models...")
    models = train_placement_models(df)
    print("✓ Models trained successfully")
    
    # Example predictions
    print("\n[3/4] Making sample predictions...")
    
    test_cases = [
        {
            'name': 'Strong Student',
            'data': {
                'cgpa': 8.5,
                'has_internship': 1,
                'internship_duration_months': 3,
                'college_tier': 1,
                'job_domain': 'IT',
                'is_metro_based': 1
            }
        },
        {
            'name': 'Average Student',
            'data': {
                'cgpa': 7.0,
                'has_internship': 1,
                'internship_duration_months': 2,
                'college_tier': 2,
                'job_domain': 'BFSI',
                'is_metro_based': 0
            }
        },
        {
            'name': 'At-Risk Student',
            'data': {
                'cgpa': 6.0,
                'has_internship': 0,
                'internship_duration_months': 0,
                'college_tier': 3,
                'job_domain': 'Manufacturing',
                'is_metro_based': 0
            }
        }
    ]
    
    print("\n" + "-"*60)
    for test_case in test_cases:
        print(f"\n📊 {test_case['name'].upper()}")
        print("-"*60)
        prediction = predict_placement(test_case['data'], models)
        
        print(f"  Placement Timeline:")
        print(f"    • 3 months:  {prediction['placement_3mo_probability']:.1f}% chance")
        print(f"    • 6 months:  {prediction['placement_6mo_probability']:.1f}% chance")
        print(f"    • 12 months: {prediction['placement_12mo_probability']:.1f}% chance")
        
        print(f"\n  Salary Prediction:")
        print(f"    • Expected:  ₹{prediction['expected_salary']:,}")
        print(f"    • Range:     ₹{prediction['salary_range_min']:,} - ₹{prediction['salary_range_max']:,}")
        
        print(f"\n  Risk Assessment:")
        print(f"    • Risk Level: {prediction['risk_level']}")
        print(f"    • Default Risk: {prediction['default_probability']:.1f}%")
        
        if prediction['strengths']:
            print(f"\n  ✅ Strengths:")
            for s in prediction['strengths']:
                print(f"    • {s}")
        
        if prediction['risks']:
            print(f"\n  ⚠️  Risks:")
            for r in prediction['risks']:
                print(f"    • {r}")
        
        print(f"\n  💡 Recommendations:")
        for rec in prediction['recommendations']:
            print(f"    • {rec}")
