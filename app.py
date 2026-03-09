import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("bankruptcy_model.pkl")

# Sidebar information
st.sidebar.title("Project Info")
st.sidebar.write("Machine Learning Bankruptcy Prediction Model")
st.sidebar.write("Model Used: Logistic Regression")
st.sidebar.write("Dataset Size: 250 companies")
st.sidebar.write("Features: 6 risk indicators")


st.title("📊 Bankruptcy Prediction App")

st.write("This application predicts the likelihood of company bankruptcy based on risk factors.")

st.write("Enter company risk values to predict bankruptcy risk.")

col1, col2 = st.columns(2)
with col1:
    st.header("Enter Company Risk Values")

    industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1])
    management_risk = st.selectbox("Management Risk", [0, 0.5, 1])
    financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1])
    credibility = st.selectbox("Credibility", [0, 0.5, 1])
    competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1])
    operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1])




# Predict button
with col2:
    if st.button("Predict Bankruptcy Risk"):

        input_data = pd.DataFrame([[industrial_risk, management_risk, financial_flexibility,
                                credibility, competitiveness, operating_risk]],
                              columns=['industrial_risk','management_risk','financial_flexibility',
                                       'credibility','competitiveness','operating_risk'])
        st.write("Input Data:")
        st.dataframe(input_data)

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.metric("Bankruptcy Risk", f"{probability*100:.1f}%")
        st.progress(int(probability * 100))

        st.subheader("📈 Prediction Result")

        if prediction[0] == 1:
            st.error("High Bankruptcy Risk Detected")
            st.warning("The company's financial indicators suggest potential bankruptcy.")

        else:
            st.success("Company Appears Financially Stable")
            st.info("Risk indicators show a low probability of bankruptcy.")

        st.write(f"Probability of Bankruptcy: **{probability:.2f}**")

        st.markdown("---")
        st.caption("Bankruptcy Prediction App | Built with Streamlit & Scikit-learn")

