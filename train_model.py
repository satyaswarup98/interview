import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("customer_feedback.csv")

# Preprocess text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
        return ' '.join(text.split())
    return ""

df['Processed_Reviews'] = df['Reviews'].apply(clean_text)

# Categorization function
def classify_text(text):
    if any(word in text for word in ["bad", "poor", "slow", "broken", "issue", "problem", "worst", "terrible"]):
        return "Complaint"
    elif any(word in text for word in ["should", "could", "better", "improve", "suggest", "recommend"]):
        return "Suggestion"
    elif any(word in text for word in ["good", "great", "excellent", "amazing", "best", "love", "happy", "awesome"]):
        return "Praise"
    return "Neutral"

df["Category"] = df["Processed_Reviews"].apply(classify_text)
df_filtered = df[df["Category"] != "Neutral"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df_filtered["Processed_Reviews"], df_filtered["Category"], test_size=0.2, random_state=42)

# Train model
model_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_pipeline, "feedback_classifier.pkl")

print("Model training complete. Accuracy:", model_pipeline.score(X_test, y_test))
