import joblib
import streamlit as st

st.set_page_config(page_title="Fake News Text Classifier", page_icon="📰")

@st.cache_resource
def load_files():
    model = joblib.load("fake_news_svm_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_files()

st.title("Fake News Text Classifier")
st.write("Paste the full text of a news article for a prediction.")

article_text = st.text_area("News article text", height=250)
word_count = len(article_text.split())
st.caption(f"Word count: {word_count}")

if st.button("Analyze text"):
    if not article_text.strip():
        st.warning("Please paste some article text first.")
    elif word_count < 100:
        st.warning(
            "Please paste at least 100 words. The model was trained mostly "
            "on full articles, so short text may lead to unreliable guesses."
        )
    else:
        text_numbers = vectorizer.transform([article_text])
        prediction = int(model.predict(text_numbers)[0])

        if prediction == 1:
            st.success("Model prediction: Likely real")
        else:
            st.error("Model prediction: Likely fake")

st.caption(
    "This student project is not a fact-checking service. "
    "Predictions can be wrong and are not proof that a story is true or false."
)
