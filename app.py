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

if st.button("Analyze text"):
    if not article_text.strip():
        st.warning("Please paste some article text first.")
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
