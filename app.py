import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake News Text Classifier",
    page_icon="📰",
    layout="centered"
)

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

st.divider()
st.subheader("Model evaluation")

st.caption(
    "Scores below come from the held-out test set of 14,427 articles. "
    "They describe test performance, not real-world fact-checking accuracy."
)

metric_columns = st.columns(3)
metric_columns[0].metric("Naive Bayes accuracy", "86.8%")
metric_columns[1].metric("Logistic Regression accuracy", "95.2%")
metric_columns[2].metric("Linear SVM accuracy", "97.6%")

st.write("Accuracy and macro F1 comparison")

chart_data = {
    "Model": ["Naive Bayes", "Logistic Regression", "Linear SVM"],
    "Accuracy (%)": [86.8, 95.2, 97.6],
    "Macro F1 (%)": [86.8, 95.2, 97.6],
}
st.bar_chart(
    chart_data,
    x="Model",
    y=["Accuracy (%)", "Macro F1 (%)"],
    sort=False,
    stack=False
)

with st.expander("View confusion matrices"):
    st.caption(
        "Rows are the actual labels; columns show the model's predictions. "
        "The label order is Fake, then Real."
    )

    matrix_columns = st.columns(3)

    matrices = [
        (
            "Naive Bayes",
            [5925, 817],
            [1081, 6604],
        ),
        (
            "Logistic Regression",
            [6603, 291],
            [403, 7130],
        ),
        (
            "Linear SVM",
            [6789, 131],
            [217, 7290],
        ),
    ]

    for column, (name, predicted_fake, predicted_real) in zip(
        matrix_columns, matrices
    ):
        with column:
            st.markdown(f"**{name}**")
            st.table({
                "Actual label": ["Fake", "Real"],
                "Predicted Fake": predicted_fake,
                "Predicted Real": predicted_real,
            })
