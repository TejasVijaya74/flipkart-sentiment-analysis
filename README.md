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
```
## Using MLflow for Experiment Tracking and Model Management

MLflow is used in this project to track experiments, compare models, and manage trained model versions.

### Features implemented
- Tracked multiple models (Logistic Regression, Linear SVM)
- Logged parameters, metrics (F1-score), and artifacts
- Compared experiment runs using MLflow UI
- Registered and versioned the best-performing model
- Enabled reproducibility and experiment history tracking

### Run training with MLflow
```bash
python scripts/train_with_mlflow.py
```
### Launch MLflow UI

### Start the MLflow tracking server:
```bash
mlflow ui
```
### Open in your browser:
```bash
http://127.0.0.1:5000
```
## Workflow Orchestration with Prefect (Optional)

Prefect is used to orchestrate the model training pipeline as a workflow.

- Pipeline steps
- Load dataset
- Preprocess text data
- Train models
- Log metrics and models to MLflow

### Run Prefect flow
```bash
python scripts/prefect_train_flow.py
```
This enables structured, reusable, and automated machine learning pipelines.
