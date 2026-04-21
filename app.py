import streamlit as st
import numpy as np
import joblib   

# = LOAD MODEL =
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Student Placement Risk & Readiness Analysis System")

# =INPUT FIELDS =

cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
tenth = st.number_input("10th Percentage", 0.0, 100.0)
twelfth = st.number_input("12th Percentage", 0.0, 100.0)
backlogs = st.number_input("Backlogs", 0, 10)

study_hours = st.number_input("Study Hours per Day", 0.0, 12.0)
attendance = st.number_input("Attendance Percentage", 0.0, 100.0)

projects = st.number_input("Projects Completed", 0, 20)
internships = st.number_input("Internships Completed", 0, 10)

coding = st.slider("Coding Skill Rating", 1, 5)
communication = st.slider("Communication Skill Rating", 1, 5)
aptitude = st.slider("Aptitude Skill Rating", 1, 5)

hackathons = st.number_input("Hackathons Participated", 0, 20)
certifications = st.number_input("Certifications Count", 0, 20)

stress = st.slider("Stress Level", 1, 10)

# ===== CATEGORICAL INPUTS =====

gender = st.selectbox("Gender", ["Male", "Female"])
branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "Mechanical", "Civil"])
part_time = st.selectbox("Part Time Job", ["Yes", "No"])
income = st.selectbox("Family Income", ["Low", "Medium", "High"])
city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
extra = st.selectbox("Extracurricular Involvement", ["Yes", "No"])

# ===== ENCODING =====

gender = 1 if gender == "Male" else 0
part_time = 1 if part_time == "Yes" else 0
extra = 1 if extra == "Yes" else 0

income = {"Low": 0, "Medium": 1, "High": 2}[income]
city = {"Tier 1": 2, "Tier 2": 1, "Tier 3": 0}[city]
branch = {"CSE": 0, "IT": 1, "ECE": 2, "Mechanical": 3, "Civil": 4}[branch]

# ===== ANALYSIS FUNCTION =====

def student_analysis(pred, readiness, risk):
    if pred == 1:
        placement = "Placed (High Probability)"
    else:
        placement = "Not Placed (Low Probability)"

    if readiness == "High":
        readiness_msg = "You are well prepared for placements."
        motivation = "Great job! Keep improving and stay consistent."
    elif readiness == "Medium":
        readiness_msg = "You are moderately prepared but need improvement."
        motivation = "Focus on skills and practice more — you can do it!"
    else:
        readiness_msg = "You are not ready yet for placements."
        motivation = "Start working on your skills — improvement is possible!"

    if risk == "Low Risk":
        risk_msg = "Your risk is low. You have good chances of placement."
    else:
        risk_msg = "Your risk is high. You need to improve your profile."

    return f"""
Placement Prediction: {placement}

Readiness Level: {readiness}
Risk Level: {risk}

Analysis:
{readiness_msg}
{risk_msg}

Motivation:
{motivation}
"""

# ===== PREDICTION =====

if st.button("Predict"):

    data = np.array([[
        gender, branch, cgpa, tenth, twelfth, backlogs,
        study_hours, attendance, projects, internships,
        coding, communication, aptitude,
        hackathons, certifications,
        stress, part_time, income, city,
        extra
    ]])

    data = scaler.transform(data)
    prediction = model.predict(data)

    # ===== RESULT =====
    if prediction[0] == 1:
        result = "Placed"
        risk = "Low Risk"
    else:
        result = "Not Placed"
        risk = "High Risk"

    # ===== READINESS =====
    score = cgpa + internships + coding

    if score >= 12:
        readiness = "High"
    elif score >= 8:
        readiness = "Medium"
    else:
        readiness = "Low"

    # ===== DISPLAY =====
    st.success(f"Prediction: {result}")
    st.info(f"Risk Level: {risk}")
    st.warning(f"Readiness Level: {readiness}")

    # ===== ANALYSIS =====
    st.text(student_analysis(prediction[0], readiness, risk))
