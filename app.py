import streamlit as st
import pandas as pd
import joblib

# Load model & data
model = joblib.load("feedback_classifier.pkl")
data = pd.read_csv("customer_segmentation.csv")

# Function to classify feedback
def classify_feedback(text):
    return model.predict([text])[0]

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.info("### Example Queries:
"
        "- What category does this feedback fall under?
"
        "- How many complaints were received last week?
"
        "- Who are the top 5 high-value customers?
"
        "- Which country has the most complaints?")

feedback = st.text_area("Enter customer feedback:")
if st.button("Classify Feedback"):
    if feedback:
        category = classify_feedback(feedback)
        st.success(f"Feedback Category: {category}")
    else:
        st.warning("Please enter feedback text.")

if st.button("Show Complaints Last Week"):
    st.write("Feature coming soon: Fetch complaints based on date from dataset.")

if st.button("Top 5 High-Value Customers"):
    data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
    top_customers = data.groupby('CustomerID')['TotalSpent'].sum().sort_values(ascending=False).head(5)
    st.write(top_customers)

if st.button("Country with Most Complaints"):
    st.write("Feature coming soon: Count complaints per country from dataset.")
