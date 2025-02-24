import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_pipeline = joblib.load("feedback_classifier.pkl")

# Load the customer segmentation data
df_segmentation = pd.read_csv("customer_segmentation.csv")

# Function to classify feedback
def classify_feedback(text):
    return model_pipeline.predict([text])[0]

# Streamlit UI
st.title("Customer Feedback Chatbot")

st.info("""### Example Queries:
- What category does this feedback fall under?
- How many complaints were received last week?
- Who are the top 5 high-value customers?
- Which country has the most complaints?
""")

# User input for feedback classification
feedback = st.text_area("Enter customer feedback:")
if st.button("Classify Feedback"):
    if feedback:
        category = classify_feedback(feedback)
        st.success(f"Feedback Category: {category}")
    else:
        st.warning("Please enter feedback text.")


# Query: Who are the top 5 high-value customers?
if st.button("Top 5 High-Value Customers"):
    df_segmentation['TotalSpent'] = df_segmentation['Quantity'] * df_segmentation['UnitPrice']
    top_customers = df_segmentation.groupby('CustomerID')['TotalSpent'].sum().sort_values(ascending=False).head(5)
    st.write(top_customers)
