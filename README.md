# Fake News Text Classifier

## What this project does

I built this project to explore whether a computer can learn patterns in news articles and use them to make a fake-or-real prediction. I also made a small Streamlit app so people can try it by pasting in an article.

## The data and how I used it

I used the WELFake dataset from [Zenodo](https://zenodo.org/records/4561253). It contains 72,134 articles with binary labels, `0` and `1`. The source materials conflict about which label corresponds to Fake or Real, so that mapping needs verification. I joined each article’s title and text, then used TF-IDF to turn the words into numbers the models could work with.

I split the data into training and test sets and compared three models. On the held-out test set of 14,427 articles, Naive Bayes reached 86.8% accuracy, Logistic Regression reached 95.2%, and Linear SVM reached 97.59% accuracy, with a macro F1 score of 97.58%. Under the notebook’s Fake/Real label convention, Linear SVM correctly identified 96.9% of examples assigned to the Fake class. The dataset’s label mapping still needs verification, so this class description should be interpreted with caution. I chose Linear SVM for the app.

These scores describe performance on this dataset. They don’t tell us whether a new article is actually true or false.

## Files

- `app.py` is the Streamlit app.
- `fake_news_svm_model.joblib` is the trained SVM model used by the app.
- `fake_news_logistic_model.joblib` is the trained Logistic Regression model.
- `tfidf_vectorizer.joblib` prepares text for the model.
- `requirements.txt` lists the packages needed to run the app.
- `Major_Project_Fake_News_Detection.ipynb` contains my analysis and training steps.

The dataset CSV isn’t included because it is large. It can be downloaded from the Zenodo link above if the model needs to be retrained.

## Running the app

Keep `app.py`, `fake_news_svm_model.joblib`, `tfidf_vectorizer.joblib`, and `requirements.txt` in the same folder. In a Windows terminal opened in that folder, run:

    py -m pip install -r requirements.txt
    py -m streamlit run app.py

## One important limitation

This is a student project, not a fact-checking service. The model can make mistakes, especially on short text or articles that are unlike the examples it learned from. Its prediction should not be treated as proof that a story is true or false.

## Report, screenshots, and live app

- [Major Project Report (PDF)](Smart_Fake_News_Detection_Project_Report.pdf)
- [Dataset Overview](SCREESHOTS/Dataset%20Overview.png)
- [Text Preprocessing](SCREESHOTS/Text%20Preprocessing.png)
- [Label Distribution](SCREESHOTS/Label%20Distribution.png)
- [Article Length Distribution](SCREESHOTS/Article%20Length%20Distribution.png)
- [Model Evaluation](SCREESHOTS/Model%20Evaluation.png)
- [App Model Dashboard](SCREESHOTS/App%20Model%20Dashboard.png)
- [App Prediction](SCREESHOTS/App%20Prediction.png)
- [Empty Input Warning](SCREESHOTS/Empty%20Input%20Warning.png)
- [Project demonstration video](https://1drv.ms/v/c/A1FD4552E3FA312F/IQAG71Sc-opEQoWj2_clkZSEAX8eY68bfIPeuU7UuZ_0-dg?e=bc4QE5)

Live app: [Open the Fake News Text Classifier](https://fakenewsdetectionclassifier-9wevyawygesyzkdvmlngbx.streamlit.app/)
