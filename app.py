import pickle
import streamlit as st

# Load the trained model
model_path = "C:/Users/Sarvadnya/Project ML/model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the vectorizer
vectorizer_path = "C:/Users/Sarvadnya/Project ML/tfidf_vectorizer.pkl"
with open(vectorizer_path, "rb") as file:
    vectorizer = pickle.load(file)

st.title("Fake News Detection App")

# User input
user_input = st.text_area("Enter a news article to check if it's Fake or Real")

if st.button("Predict"):
    if user_input:
        # Transform the input text
        transformed_text = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 0:
            st.write("🛑 This news is **Fake** ❌")
        else:
            st.write("✅ This news is **Real** 📰")
    else:
        st.warning("Please enter some text for prediction.")
