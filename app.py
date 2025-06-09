import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Campus Placement Prediction System")

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])
p1 = 1 if gender == "Male" else 0

# SSC Percentage
p2 = st.number_input("Secondary Education percentage - 10th Grade", 0.0, 100.0)

# SSC Board
ssc_board = st.selectbox("Board of Education (SSC)", ["Central", "Others"])
p3 = 1 if ssc_board == "Central" else 0

# HSC Percentage
p4 = st.number_input("Higher Secondary Education percentage - 12th Grade", 0.0, 100.0)

# HSC Board
hsc_board = st.selectbox("Board of Education (HSC)", ["Central", "Others"])
p5 = 1 if hsc_board == "Central" else 0

# HSC Specialization
hsc_s = st.selectbox("Specialization in Higher Secondary Education", ["Science", "Commerce", "Arts"])
p6 = {"Arts": 0, "Commerce": 1, "Science": 2}[hsc_s]

# Degree Percentage
p7 = st.number_input("Degree Percentage", 0.0, 100.0)

# Degree Type
degree_t = st.selectbox("Field of Degree Education", ["Others", "Comm&Mgmt", "Sci&Tech"])
p8 = {"Others": 0, "Comm&Mgmt": 1, "Sci&Tech": 2}[degree_t]

# Work Experience
workex = st.selectbox("Work Experience", ["Yes", "No"])
p9 = 1 if workex == "Yes" else 0

# E-test Percentage
p10 = st.number_input("Enter Test Percentage", 0.0, 100.0)

# MBA Specialisation
spec = st.selectbox("Branch Specialisation", ["Mky&Fin", "Mkt&HR"])
p11 = 1 if spec == "Mkt&HR" else 0

# MBA Percentage
p12 = st.number_input("MBA Percentage", 0.0, 100.0)

#Hackathon participation
Hackathon = st.selectbox("Hackathon participation", ["Yes", "No"])
p13 = 1 if Hackathon == "Yes" else 0

#aptitude percentage
p14 = st.number_input("aptitude Percentage", 0.0, 100.0)

#coding marks
p15 = st.number_input("coding marks", 0.0, 200.0)

# Predict Button
if st.button("Predict Placement"):
    model = joblib.load("model_placement1")
    new_data = pd.DataFrame([{
        'gender': p1,
        'ssc_p': p2,
        'ssc_b': p3,
        'hsc_p': p4,
        'hsc_b': p5,
        'hsc_s': p6,
        'degree_p': p7,
        'degree_t': p8,
        'workex': p9,
        'etest_p': p10,
        'specialisation': p11,
        'mba_p': p12,
        'Hackathon': p13,
        'aptitude_score': p14,
        'coding_score': p15
    }])

    result = model.predict(new_data)
    result1 = model.predict_proba(new_data)

    if result[0] == 0:
        st.error("❌ Sorry! The student is likely *not* to be placed.")
    else:
        probability = round(result1[0][1] * 100, 2)
        st.success(f"✅ The student is likely to be placed with a probability of **{probability}%**.")
