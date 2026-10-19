import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Text Classifier", page_icon="📰")

@st.cache_resource
def load_files():
    model = joblib.load("fake_news_logistic_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_files()

st.title("Fake News Text Classifier")
st.write("Paste a news headline or article to see the model's prediction.")

article_text = st.text_area("News text", height=250)

if st.button("Analyze text"):
    if not article_text.strip():
        st.warning("Please paste some text first.")
    else:
        text_numbers = vectorizer.transform([article_text])
        prediction = int(model.predict(text_numbers)[0])

        if prediction == 1:
            st.success("Model prediction: Likely real")
        else:
            st.error("Model prediction: Likely fake")

st.caption(
    "This is a student project. Its prediction is not a fact-check "
    "and should not be used as proof that a news story is true or false."
)
