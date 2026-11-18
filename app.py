import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Maternal Risk Dashboard", page_icon="🤰")

@st.cache_resource
def load_model():
    df = pd.read_csv("Maternal Health Risk Data Set.csv", sep="\t")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df["HeartRate"] >= 40].reset_index(drop=True)
    X = df.drop("RiskLevel", axis=1)
    y = df["RiskLevel"]
    m = GradientBoostingClassifier(
        learning_rate=0.05, max_depth=2, n_estimators=100, random_state=42)
    m.fit(X, y)
    return m

model = load_model()

st.title("🤰 Maternal Health Risk Predictor")
st.write("Mother ki details daalo aur risk level dekho.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (saal)", 10, 70, 28)
    systolic = st.number_input("Systolic BP", 70, 200, 120)
    diastolic = st.number_input("Diastolic BP", 40, 120, 80)
with col2:
    bs = st.number_input("Blood Sugar (BS)", 3.0, 25.0, 7.0, step=0.1)
    temp = st.number_input("Body Temp (°F)", 95.0, 105.0, 98.0, step=0.1)
    hr = st.number_input("Heart Rate (bpm)", 40, 130, 75)

if st.button("Risk Check Karo"):
    data = pd.DataFrame([{
        "Age": age, "SystolicBP": systolic, "DiastolicBP": diastolic,
        "BS": bs, "BodyTemp": temp, "HeartRate": hr
    }])
    pred = model.predict(data)[0]
    probs = model.predict_proba(data)[0]

    if pred == "high risk":
        st.error("🔴 HIGH RISK - turant doctor se milein")
    elif pred == "mid risk":
        st.warning("🟠 MID RISK - doctor se salah lein")
    else:
        st.success("🟢 LOW RISK")

    st.subheader("Probability")
    prob_df = pd.DataFrame({"Risk": model.classes_, "Probability": probs}).set_index("Risk")
    st.bar_chart(prob_df)

    if pred != "high risk" and probs[list(model.classes_).index("high risk")] > 0.25:
        st.warning("⚠️ High risk ki sambhavna kaafi hai. Doctor se zaroor milein.")

st.caption("Ye sirf ek educational tool hai. Doctor ki salah ka vikalp nahi.")