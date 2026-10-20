# Fake News Text Classifier

## What this project does

I built this project to explore whether a computer can learn patterns in news articles and use them to make a fake-or-real prediction. I also made a small Streamlit app so people can try it by pasting in an article.

## The data and how I used it

I used the WELFake dataset from [Zenodo](https://zenodo.org/records/4561253). It contains 72,134 labeled articles: `0` means fake and `1` means real. I joined each article’s title and text, then used TF-IDF to turn the words into numbers the models could work with.

I split the data into training and test sets and compared three models. On the held-out test set of 14,427 articles, Naive Bayes reached 86.8% accuracy, Logistic Regression reached 95.2%, and Linear SVM reached 97.6% accuracy, with a macro F1 score of 97.6%. The Linear SVM correctly identified 96.9% of the fake articles in that test set, so I chose it for the app.

These scores describe performance on this dataset. They don’t tell us whether a new article is actually true or false.

## Files

- `app.py` is the app people can use.
- `fake_news_svm_model.joblib` is the trained SVM model.
- `tfidf_vectorizer.joblib` prepares text for the model.
- `requirements.txt` lists the packages needed to run the app.
- `Major_Project_Fake_News_Detection.ipynb` contains my analysis and training steps.

The dataset CSV isn’t included because it is large. It can be downloaded from the Zenodo link above if the model needs to be retrained.

## Running the app

Keep `app.py`, both `.joblib` files, and `requirements.txt` in the same folder. In a Windows terminal opened in that folder, run:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

## One important limitation

This is a student project, not a fact-checking service. The model can make mistakes, especially on short text or articles that are unlike the examples it learned from. Its prediction should not be treated as proof that a story is true or false.
## Report, screenshots, and live app

- [Major Project Report (PDF)](Fake_News_Detection_Major_Project_Report.pdf)
- [Dataset Overview](SCREENSHOTS/Dataset%20Overview.png)
- [Text Preprocessing](SCREENSHOTS/Text%20Preprocessing.png)
- [Label Distribution](SCREENSHOTS/Label%20Distribution.png)
- [Article Length Distribution](SCREENSHOTS/Article%20Length%20Distribution.png)
- [Model Evaluation](SCREENSHOTS/Model%20Evaluation.png)
- [App Model Dashboard](SCREENSHOTS/App%20Model%20Dashboard.png)
- [App Prediction](SCREENSHOTS/App%20Prediction.png)
- [Empty Input Warning](SCREENSHOTS/Empty%20Input%20Warning.png)

Live app: [Open the Fake News Text Classifier](https://fakenewsdetectionclassifier-9wevyawygesyzkdvmlngbx.streamlit.app/)
