import streamlit as st
import numpy as np
import joblib
import requests

st.set_page_config(page_title="Credit Score Simulator", layout="centered")
st.title("💳 Credit Score Simulator")
st.markdown("Predict your credit score and get AI advice based on your financial info.")

# --- Load model components ---
try:
    model = joblib.load("credit_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# --- Manual encoding maps ---
enc_mix = {'Bad': 0, 'Good': 1, 'Standard': 2}
enc_minpay = {'No': 0, 'Unknown': 1, 'Yes': 2}
enc_behave = {
    'High_spent_Large_value_payments': 0,
    'High_spent_Medium_value_payments': 1,
    'Low_spent_Large_value_payments': 2,
    'Low_spent_Small_value_payments': 3
}
enc_occupation = {
    'Accountant': 0, 'Engineer': 1, 'Doctor': 2, 'Lawyer': 3,
    'Teacher': 4, 'Entrepreneur': 5, 'Manager': 6, 'Scientist': 7,
    'Developer': 8, 'Unknown': 9
}

# --- User Input ---
with st.form("form"):
    st.subheader("📋 Enter your financial details:")
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 0, 500000, 50000)
    inhand_salary = st.number_input("Monthly Inhand Salary ($)", 0, 20000, 3000)
    bank_accounts = st.slider("Number of Bank Accounts", 0, 10, 2)
    credit_cards = st.slider("Number of Credit Cards", 0, 10, 1)
    interest_rate = st.slider("Interest Rate (%)", 0, 40, 10)
    loans = st.slider("Number of Loans", 0, 10, 1)
    delayed_payments = st.slider("Delayed Payments", 0, 20, 0)
    credit_limit_change = st.number_input("Change in Credit Limit ($)", value=0.0)
    credit_inquiries = st.slider("Credit Inquiries", 0, 15, 1)
    credit_mix = st.selectbox("Credit Mix", list(enc_mix.keys()))
    occupation = st.selectbox("Occupation", list(enc_occupation.keys()))
    debt = st.number_input("Outstanding Debt ($)", 0.0, 100000.0, 5000.0)
    utilization = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 35.0)
    history = st.slider("Credit History Age (months)", 0, 480, 60)
    min_payment = st.selectbox("Payment of Minimum Amount", list(enc_minpay.keys()))
    emi = st.number_input("EMI Per Month ($)", 0.0, 10000.0, 150.0)
    invested = st.number_input("Invested Monthly ($)", 0.0, 10000.0, 200.0)
    behaviour = st.selectbox("Payment Behaviour", list(enc_behave.keys()))
    balance = st.number_input("Monthly Balance ($)", 0.0, 10000.0, 500.0)
    changed_limit = st.number_input("Changed Credit Limit ($)", 0.0, 10000.0, 100.0)
    submit = st.form_submit_button("🔮 Predict")

# --- Run Prediction ---
if submit:
    try:
        mix = enc_mix.get(credit_mix, 1)
        minpay = enc_minpay.get(min_payment, 2)
        behave = enc_behave.get(behaviour, 3)
        occup = enc_occupation.get(occupation, 9)

        inputs = np.array([
            age, income, inhand_salary, bank_accounts, credit_cards,
            interest_rate, loans, delayed_payments, credit_limit_change,
            credit_inquiries, mix, occup, debt, utilization, history,
            minpay, emi, invested, behave, balance, changed_limit
        ]).reshape(1, -1)

        scaled = scaler.transform(inputs)
        pred = model.predict(scaled)[0]

        score_map = {0: "Poor", 1: "Standard", 2: "Good"}
        score_label = score_map.get(pred, "Unknown")
        st.subheader(f"🔿 Your Predicted Credit Score: {score_label}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # --- Together AI Advice ---
    st.markdown("### 🤖 Credit Advice from AI")
    try:
        headers = {
            "Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}",
            "Content-Type": "application/json"
        }

        prompt = f"""
        User Info: Age={age}, Income={income}, Credit Cards={credit_cards}, Delayed Payments={delayed_payments},
        Debt={debt}, Utilization={utilization}, Credit Mix={credit_mix}, Occupation={occupation}, Score={score_label}.
        Provide personalized credit score improvement advice.
        """

        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95
        }

        res = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)
        if res.status_code == 200:
            suggestion = res.json()["choices"][0]["text"]
            st.success(suggestion.strip())
        else:
            st.warning("Together AI could not respond at this time.")
    except Exception as e:
        st.warning(f"AI advice failed: {e}")


