# ui/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hospitality Insights", layout="wide")

st.title("📊 Hospitality Insights Dashboard")
st.markdown("A simple Streamlit app for exploring data and model outputs.")

# --- Load or simulate data ---
data = {
    "Review": ["Great stay", "Average breakfast", "Noisy room", "Excellent service"],
    "Sentiment": ["Positive", "Neutral", "Negative", "Positive"],
    "Score": [0.95, 0.65, 0.30, 0.98],
}
df = pd.DataFrame(data)

# --- Display table ---
st.subheader("Customer Reviews")
st.dataframe(df)

# --- Visualize sentiment scores ---
fig = px.bar(df, x="Review", y="Score", color="Sentiment", title="Sentiment Confidence")
st.plotly_chart(fig, use_container_width=True)
