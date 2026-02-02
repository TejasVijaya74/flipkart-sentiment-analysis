# Flipkart Review Sentiment Analysis

## Overview
This project performs sentiment analysis on Flipkart product reviews to classify customer feedback as **Positive** or **Negative** using machine learning.

The model is trained on pre-scraped review data and deployed as a **real-time Streamlit web application**.

---

## Technologies
Python
Scikit-learn
NLTK
Streamlit
AWS EC2

---

## Dataset
- Source: Flipkart product reviews (provided dataset)
- Features used:
  - Rating
  - Review Title
  - Review Text
- Neutral reviews (rating = 3) are excluded

---

## Approach
1. Text preprocessing (cleaning, stopwords removal, lemmatization)
2. Feature extraction using **TF-IDF**
3. Model training using Logistic Regression and Linear SVM
4. Model evaluation using **F1-score**
5. Deployment using **Streamlit**

---

## Live Application
The application is deployed on AWS EC2:
http://51.20.85.203:8501


---

## Run Locally
```bash
git clone https://github.com/TejasVijaya74/flipkart-sentiment-analysis.git
cd flipkart-sentiment-analysis
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r environment.txt
streamlit run app/app.py
