# Fake News Text Classifier

## About this project

For this project, I built a simple app that uses machine learning to look for patterns in news articles and predict whether an article is likely to be fake or real. The app lets someone paste in a headline or article and view the model’s prediction.

## The data

I used the WELFake dataset, which contains 72,134 news articles. Each article has a label: `0` means fake and `1` means real. The dataset is available on [Zenodo](https://zenodo.org/records/4561253).

The article titles and text were combined before training. I kept the dataset CSV out of this project folder because it is large. The download link above can be used to get it again.

## How I built it

First, I checked the data for missing values and duplicates. Then I used TF-IDF to turn the article text into numbers that machine-learning models can work with.

I trained and compared two models using an 80/20 split: most articles were used for training, and the rest were held back to evaluate the models.

- Naive Bayes achieved **86.8% accuracy** on the test data.
- Logistic Regression achieved **95.2% accuracy** on the test data.

I chose Logistic Regression for the app because it performed better on this test set.

## Files in this project

- `app.py` — the app for trying a prediction
- `fake_news_logistic_model.joblib` — the trained model
- `tfidf_vectorizer.joblib` — the tool that converts text into model-ready numbers
- `requirements.txt` — the Python packages needed to run the app
- `Major_Project_Fake_News_Detection.ipynb` — the notebook with the data analysis and model training

## Running the app

Keep `app.py`, both `.joblib` files, and `requirements.txt` together in one folder. On Windows, open a terminal in that folder and run:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

## A note about the predictions

This project is a learning demonstration, not a fact-checking service. The model can make mistakes, so its prediction should not be treated as proof that an article is true or false.
