import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

print("Starting training script...")

mlflow.set_experiment("Flipkart_Sentiment_Analysis")

print("Loading dataset...")

df = pd.read_csv("data/processed/cleaned_reviews.csv")
print("Dataset loaded:", df.shape)

X = df["clean_review"].fillna("")
y = df["sentiment"]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Vectorizing text...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Starting MLflow run: Logistic Regression")

with mlflow.start_run(run_name="Logistic_Regression"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

print("Starting MLflow run: Linear SVM")

with mlflow.start_run(run_name="Linear_SVM"):
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model", "LinearSVM")
    mlflow.log_param("max_features", 5000)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

print("Training completed successfully.")
