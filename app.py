import streamlit as st
import pandas as pd
import joblib

# Load model & data
model = joblib.load("feedback_classifier.pkl")
data = pd.read_csv("customer_segmentation.csv")

# Function to classify feedback
def classify_feedback(text):
    return model.predict([text])[0]

# Function to process user queries
def process_query(query):
    query = query.lower()

    if "category" in query or "feedback" in query:
        return "Please enter the feedback you want to classify."

    elif "top customers" in query or "high-value customers" in query:
        data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
        top_customers = data.groupby('CustomerID')['TotalSpent'].sum().sort_values(ascending=False).head(5)
        return "Top 5 High-Value Customers:\n" + top_customers.to_string()

    elif "complaints last week" in query:
        return "Feature coming soon: Fetch complaints based on date from dataset."

    elif "most complaints" in query or "which country" in query:
        return "Feature coming soon: Count complaints per country from dataset."

    else:
        return "Sorry, I didn't understand the question. Try asking about feedback categories, complaints, or customer insights."

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.info(
    "### Example Queries:\n"
    "- What category does this feedback fall under?\n"
    "- How many complaints were received last week?\n"
    "- Who are the top 5 high-value customers?\n"
    "- Which country has the most complaints?\n"
)

# User input for open-ended chatbot interaction
user_query = st.text_input("Ask me a question:")

if user_query:
    response = process_query(user_query)
    st.write(response)

    # If user asks about feedback classification, allow them to input feedback
    if "category" in user_query or "feedback" in user_query:
        feedback = st.text_area("Enter your feedback:")
        if feedback:
            category = classify_feedback(feedback)
            st.success(f"Feedback Category: {category}")
